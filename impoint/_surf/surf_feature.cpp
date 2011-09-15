#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "SurfDetect.h"
#include "SurfDescribe.h"
#include "SurfMatch.h"
#include "surf_feature.hpp"

bool cmp_surf_point(SurfPoint* s0, SurfPoint* s1) {
  if (s0->cornerness > s1->cornerness)
    return true;
  return false;
}

int compute_surf_descriptors(char *data, int height, int width, int max_points, float *points) {
  int total_points = 0;
  IntegralImage intim(data, height, width);
  std::list<SurfPoint*> *sp = SurfDetect::allocPoints();
  SurfDetect sdet(height, width);
  sdet.compute(sp, &intim);
  SurfDescribe *sdesc = new SurfDescribe();
  sdesc->compute(sp, &intim);
  int feature_bytes = sizeof(float) * 64;
  sp->sort(cmp_surf_point);
  for (std::list<SurfPoint*>::iterator iter = sp->begin(); iter != sp->end(); ++iter, ++total_points) {
    if (total_points >= max_points)
      break;
    memcpy(points, (*iter)->features64, feature_bytes);
    points += 64;
  }
  delete sdesc;
  SurfDetect::freePoints(&sp);
  return total_points;
}

int compute_surf_points(char *data, int height, int width, int max_points, float *points, int *x, int *y, int *scale, float *orientation, char *sign, float *cornerness) {
  int total_points = 0;
  IntegralImage intim(data, height, width);
  std::list<SurfPoint*> *sp = SurfDetect::allocPoints();
  SurfDetect sdet(height, width);
  sdet.compute(sp, &intim);
  SurfDescribe *sdesc = new SurfDescribe();
  sdesc->compute(sp, &intim);
  int feature_bytes = sizeof(float) * 64;
  sp->sort(cmp_surf_point);
  for (std::list<SurfPoint*>::iterator iter = sp->begin(); iter != sp->end(); ++iter, ++total_points) {
    if (total_points >= max_points)
      break;
    memcpy(points, (*iter)->features64, feature_bytes);
    points += 64;
    *(x++) = (*iter)->x;
    *(y++) = (*iter)->y;
    *(scale++) = (*iter)->scale;
    *(orientation++) = (*iter)->orientation;
    *(sign++) = (*iter)->sign;
    *(cornerness++) = (*iter)->cornerness;
  }
  delete sdesc;
  SurfDetect::freePoints(&sp);
  return total_points;
}

void compute_descriptors(char *data, int height, int width, int (*feat_callback)(int *, int *, int *), void (*collect_callback)(float *)) {
    IntegralImage intim(data, height, width);
    SurfPoint *sp = new SurfPoint(0, 0, 0, false, 0.);
    SurfDescribe *sdesc = new SurfDescribe();
    int x, y, scale;
    while(feat_callback(&x, &y, &scale)) {
        sp->x = x;
        sp->y = y;
        sp->scale = scale;
        sdesc->compute(sp, &intim);
        collect_callback(sp->features64);
    }
    delete sdesc;
    delete sp;
}

void convert_points(std::list<SurfPoint*> *sps, float *features, int *x, int *y, int *scale, float *orientation, char *sign, float *cornerness, int num_points, int is64) {
    int i;
    for (i = 0; i < num_points; ++i) {
        SurfPoint *sp = new SurfPoint(x[i], y[i], scale[i], sign[i], cornerness[i]);
        sp->index = i;
        if (features) {
            if (is64) {
                sp->features64 = new float[64];
                memcpy(sp->features64, features, sizeof(float) * 64);
                features += 64;
            } else {
                sp->features128 = new float[128];
                memcpy(sp->features128, features, sizeof(float) * 128);
                features += 128;
            }
        }
        sps->push_back(sp);
    }
}

int match_surf_points(float *features0, int *x0, int *y0, int *scale0, float *orientation0,
                      char *sign0, float *cornerness0, int num_points0,
                      float *features1, int *x1, int *y1, int *scale1, float *orientation1,
                      char *sign1, float *cornerness1, int num_points1,
                      int is64, float threshNNRD, float threshNND,
                      int **out_matches) {
  std::list<SurfPoint*> *sp0 = SurfDetect::allocPoints();
  std::list<SurfPoint*> *sp1 = SurfDetect::allocPoints();
  convert_points(sp0, features0, x0, y0, scale0, orientation0, sign0, cornerness0, num_points0, is64);
  convert_points(sp1, features1, x1, y1, scale1, orientation1, sign1, cornerness1, num_points1, is64);
  SurfMatch *sm = new SurfMatch();
  std::list<MatchPair*> *matchList = SurfMatch::allocMatches();
  sm->matchNNDR(matchList, sp0, sp1, threshNNRD, threshNND);
  int num_matches = matchList->size();
  int *matches = (int *)malloc(sizeof(int) * num_matches * 2); // NOTE: Using malloc so that C code can free it
  *out_matches = matches;
  for (list<MatchPair*>::iterator mp = matchList->begin(); mp != matchList->end(); mp++) {
      matches[0] = (*mp)->spt0->index;
      matches[1] = (*mp)->spt1->index;
      matches += 2;
  }
  SurfMatch::freeMatchList(&matchList);
  SurfDetect::freePoints(&sp0);
  SurfDetect::freePoints(&sp1);
  delete sm;
  return num_matches;
}
