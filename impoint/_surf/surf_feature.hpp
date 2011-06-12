#ifdef __cplusplus
extern "C" {
#endif
  int compute_surf_descriptors(char *data, int height, int width, int max_points, float *points);
  int compute_surf_points(char *data, int height, int width, int max_points, float *points, int *x, int *y, int *scale, float *orientation, char *sign, float *cornerness);
  int match_surf_points(float *features0, int *x0, int *y0, int *scale0, float *orientation0,
                        char *sign0, float *cornerness0, int num_points0,
                        float *features1, int *x1, int *y1, int *scale1, float *orientation1,
                        char *sign1, float *cornerness1, int num_points1,
                        int is64, float threshNNRD, float threshNND,
                        int **out_matches);
#ifdef __cplusplus
}
#endif
