#include <float.h>
#include "SurfMatch.h"
#include <cstdio>
#include <cmath>
//#include "common_macros.h"

SurfMatch::SurfMatch()
{
#ifdef STATS
	dimQuit = new int[64];
	for (int i = 0; i < 64; i++) {
		dimQuit[i] = 0;
	}
#endif
}

SurfMatch::~SurfMatch()
{
#ifdef STATS
	int total=0;
	for (int i = 0; i < 64; i++) {
		if (dimQuit[i]) {
			printf("%d:%d ",i,dimQuit[i]);
			total+=dimQuit[i];
		}
	}
	printf("\n");
	printf("TotalDumpedMatches:%d\n",total);
	delete [] dimQuit;
#endif
}



void SurfMatch::selfNND(list<SurfPoint*> *points, float threshNND)
{
	list<SurfPoint*> *badPoints = new list<SurfPoint*>();
	for (list<SurfPoint*>::iterator pt0 = points->begin(); pt0 != points->end(); pt0++) {
		list<SurfPoint*>::iterator pt1 = pt0;
		pt1++;
		while (pt1 != points->end()) { // TODO Go the next value first
			if ((*pt0)->sign != (*pt1)->sign) {
				pt1++;
				continue;
			}
			float dist = matchNNDDist64(*pt0,*pt1);
			if (dist < threshNND) {
				badPoints->push_back(*pt0);
				break;
			}
			pt1++;
		}
	}

	// Delete bad points
	for (list<SurfPoint*>::iterator pt = badPoints->begin(); pt != badPoints->end(); pt++) {
		delete *pt;
		points->remove(*pt); // TODO Should not be using remove!
	}
	delete badPoints;
}

bool SurfMatch::matchNNDR(list<MatchPair*> *matchList, list<SurfPoint*> *points0, list<SurfPoint*> *points1, float threshNNRD)
{
	return matchNNDR(matchList, points0, points1, threshNNRD, FLT_MAX);
}

bool SurfMatch::matchNNDR(list<MatchPair*> *matchList, list<SurfPoint*> *points0, list<SurfPoint*> *points1, float threshNNRD, float threshNND)
{
	return matchNNDR(matchList, points0, points1, threshNNRD, threshNND,FLT_MAX);
}

// For each point M_0 in im0 and M_1 in im1, we need to find the minimum 2 distances between then, find their ratio,
// and add them to the matchList if they pass some thresholds.
bool SurfMatch::matchNNDR(list<MatchPair*> *matchList, list<SurfPoint*> *points0, list<SurfPoint*> *points1, float threshNNRD, float threshNND, float threshD128)
{
	int outerCnt = 0;
	bool anyValid = false;
	double threshNNDDiv8 = threshNND/32.0;
	for (list<SurfPoint*>::iterator pt0 = points0->begin(); pt0 != points0->end(); pt0++) {
		SurfPoint* pt1N0 = 0;
		float pt1N0Dist = FLT_MAX;
		SurfPoint* pt1N1 = 0;
		float pt1N1Dist = FLT_MAX;
		bool skip128 = threshD128 == FLT_MAX;
		//if (outerCnt >= 250 && !anyValid)
		//  return false;
		outerCnt++;
		for (list<SurfPoint*>::iterator pt1 = points1->begin(); pt1 != points1->end(); pt1++) {
			// If the signs differ then don't attempt to match them
			if ((*pt0)->sign != (*pt1)->sign)
				continue;

			float dist = matchNNDDist64(*pt0,*pt1);
			//float dist = matchNNDDist64Approx(*pt0,*pt1,threshNNDDiv8);
			if (dist < pt1N0Dist) {
				pt1N1Dist = pt1N0Dist;
				pt1N1 = pt1N0;
				pt1N0Dist = dist;
				pt1N0 = *pt1;
			} else if (dist < pt1N1Dist) {
				pt1N1Dist = dist;
				pt1N1 = *pt1;
			}
		}
		// If there are no distances
		if (!pt1N0)
			continue;
		// If there is only one distance
		if (!pt1N1 && pt1N0Dist < threshNND) {
			matchList->push_back(new MatchPair(*pt0,pt1N0));
			continue;
		}
		// If there is more than one distance
		if (pt1N0Dist/pt1N1Dist < threshNNRD && pt1N0Dist < threshNND) {
			// Use the 128 dimension
			if (skip128 || matchNNDDist128(*pt0,pt1N0) < threshD128) {
				anyValid = true;
				matchList->push_back(new MatchPair(*pt0,pt1N0));
			}
		}
	}
	return true;
}

float SurfMatch::matchNNDDist64(SurfPoint *pt0, SurfPoint *pt1, float threshNND)
{
	float sum = 0.0f;

	for (int i = 0; i < 64; i++) {
		float diff = pt1->features64[i] - pt0->features64[i];
		sum += diff*diff;
		if (sum >= threshNND) {
#ifdef STATS
			dimQuit[i]++;
#endif
			return sum;
		}
	}
	return sum;
}

float SurfMatch::matchNNDDist64Approx(SurfPoint *pt0, SurfPoint *pt1, float threshNND)
{
	float sum = 0.0f;

	for (int i = 0; i < 8; i++) {
		float diff = pt1->features64[i] - pt0->features64[i];
		sum += diff*diff;
	}
	if (sum >= threshNND)
		return FLT_MAX;

	for (int i = 8; i < 64; i++) {
		float diff = pt1->features64[i] - pt0->features64[i];
		sum += diff*diff;
	}

	return sum;
}

float SurfMatch::matchNNDDist64Stats(SurfPoint *pt0, SurfPoint *pt1, float threshNND)
{
	float sum = 0.0f;
	int i=0;
	for (int k=0; k < 4; k++) {
		for (int j = 0; j < 16; j++) {
			float diff = pt1->features64[i] - pt0->features64[i];
			sum += diff*diff;
			i++;
		}
		if (sum >= threshNND) {
#ifdef STATS
			dimQuit[i]++;
#endif
			return sum;
		}
	}
	return sum;
}

float SurfMatch::matchNNDDist64(SurfPoint *pt0, SurfPoint *pt1)
{
	float sum = 0.0f;

	for (int i = 0; i < 64; i++) {
		float diff = pt1->features64[i] - pt0->features64[i];
		sum += diff*diff;
	}
	return sum;
}


float SurfMatch::matchNNDDist64L1(SurfPoint *pt0, SurfPoint *pt1)
{
	float sum = 0.0f;

	for (int i = 0; i < 64; i++) {
		float diff = pt1->features64[i] - pt0->features64[i];
		sum += abs(diff);
	}
	return sum;
}

float SurfMatch::matchNNDDist128(SurfPoint *pt0, SurfPoint *pt1)
{
	float sum = 0.0f;

	for (int i = 0; i < 128; i++) {
		float diff = pt1->features128[i] - pt0->features128[i];
		sum += diff*diff;
	}
	return sum;
}

void SurfMatch::printMatchList(list<MatchPair*> *matchList)
{
	printMatchList(stdout, matchList);
}

void SurfMatch::printMatchList(FILE* fp, list<MatchPair*> *matchList)
{
	float mag,xDiff,yDiff;

	for (list<MatchPair*>::iterator mp = matchList->begin(); mp != matchList->end(); mp++) {
		xDiff=(*mp)->pt0[0]-(*mp)->pt1[0];
		yDiff=(*mp)->pt0[1]-(*mp)->pt1[1];
		mag = sqrt(xDiff*xDiff+yDiff*yDiff);
		fprintf(fp,"%f %f %f %f\n",(*mp)->pt0[0], (*mp)->pt0[1], (*mp)->pt1[0], (*mp)->pt1[1]);
	}
}
