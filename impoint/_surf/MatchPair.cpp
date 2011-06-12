#include "MatchPair.h"
MatchPair::MatchPair(SurfPoint* pt0, SurfPoint* pt1)
{
	this->pt0 = new float[2];
	this->pt1 = new float[2];
	this->pt0[0] = pt0->x;
	this->pt0[1] = pt0->y;
	this->pt1[0] = pt1->x;
	this->pt1[1] = pt1->y;
	spt0 = pt0;
	spt1 = pt1;
}
MatchPair::MatchPair()
{
	pt0 = new float[2];
	pt1 = new float[2];
	spt0 = 0;
	spt1 = 0;
}

MatchPair::MatchPair(float x0, float y0, float x1, float y1)
{
	pt0 = new float[2];
	pt0[0]=x0;
	pt0[1]=y0;
	pt1 = new float[2];
	pt1[0]=x1;
	pt1[1]=y1;
	spt0 = 0;
	spt1 = 0;
}

MatchPair::MatchPair(float *pt0, float *pt1)
{
	this->pt0 = new float[2];
	this->pt0[0]=pt0[0];
	this->pt0[1]=pt0[1];
	this->pt1 = new float[2];
	this->pt1[0]=pt1[0];
	this->pt1[1]=pt1[1];
	spt0 = 0;
	spt1 = 0;
}

MatchPair::~MatchPair()
{
	delete [] pt0;
	pt0 = 0;
	delete [] pt1;
	pt1 = 0;
}
