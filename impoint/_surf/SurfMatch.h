#include <list>
#include <cstdio>
#include "SurfPoint.h"
#include "MatchPair.h"
using namespace std;
class SurfMatch
{
public:
	SurfMatch();
	~SurfMatch();
	bool matchNNDR(std::list<MatchPair*> *matchList, std::list<SurfPoint*> *points0, std::list<SurfPoint*> *points1, float threshNNDR);
	bool matchNNDR(std::list<MatchPair*> *matchList, std::list<SurfPoint*> *points0, std::list<SurfPoint*> *points1, float threshNNRD, float threshNND);
	bool matchNNDR(std::list<MatchPair*> *matchList, std::list<SurfPoint*> *points0, std::list<SurfPoint*> *points1, float threshNNRD, float threshNND, float threshD128);
	void selfNND(std::list<SurfPoint*> *points, float threshNND);
	float matchNNDDist64(SurfPoint *pt0, SurfPoint *pt1);
	float matchNNDDist64(SurfPoint *pt0, SurfPoint *pt1, float threshNND);
	float matchNNDDist64Approx(SurfPoint *pt0, SurfPoint *pt1, float threshNND);
	float matchNNDDist64Stats(SurfPoint *pt0, SurfPoint *pt1, float threshNND);
	float matchNNDDist64L1(SurfPoint *pt0, SurfPoint *pt1);


	float matchNNDDist128(SurfPoint *pt0, SurfPoint *pt1);
	void printMatchList(std::list<MatchPair*> *matchList);
	void printMatchList(FILE* fp, std::list<MatchPair*> *matchList);

	static std::list<MatchPair*> *allocMatches() {
		return new list<MatchPair*>();
	}

	static void freeMatches(std::list<MatchPair*> **matchList) {
		for (list<MatchPair*>::iterator mp = (*matchList)->begin(); mp != (*matchList)->end(); mp++) {
			delete *mp;
		}
		delete *matchList;
		*matchList = 0;
	}

	static void freeMatches(std::list<MatchPair*> *matchList) {
	  freeMatches(&matchList);
	}

	static void freeMatchList(std::list<MatchPair*> **matchList) {
		delete *matchList;
		*matchList = 0;
	}

	static void freeMatchList(std::list<MatchPair*> *matchList) {
	  freeMatchList(&matchList);
	}

private:
#ifdef STATS
	int *dimQuit;
#endif
};
