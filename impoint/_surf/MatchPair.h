#ifndef MATCHPAIR_H
#define MATCHPAIR_H
#include "SurfPoint.h"

class MatchPair
{
public:
	MatchPair(SurfPoint* pt0, SurfPoint* pt1);
	MatchPair(float x0, float y0, float x1, float y1);
	MatchPair();
	MatchPair(float *pt0, float *pt1);
	virtual ~MatchPair();
	float *pt0;
	float *pt1;
	SurfPoint *spt0;
	SurfPoint *spt1;
};
#endif
