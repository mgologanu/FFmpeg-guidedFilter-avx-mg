/*
 * Copyright (c) 2021 Xuewei Meng
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "filters.h"
#include "framesync.h"
#include "video.h"

#include<immintrin.h>
#include <stdalign.h>
#include <stdbool.h> 
#include <stddef.h> // For size_t
#include <stdint.h>
#include <string.h>
#include <stdio.h>


enum FilterModes {
    BASIC,
    FAST,
    NB_MODES,
};

enum GuidanceModes {
    OFF,
    ON,
    NB_GUIDANCE_MODES,
};

typedef struct GuidedContext {
  const AVClass *class;
  FFFrameSync fs;
  
  int radius;
  float eps;
  int mode;
  int sub;
  int guidance;
  int planes;
  
  int width;
  int height;
  
  int nb_planes;
  int depth;
  int planewidth[4];
  int planeheight[4];
  
  float *I;
  float *p;
  float *work;

  float *t1;
  float *t2;
  float *t3;
  float *ai;
  float *bi;

  float *ai2;
  
  int (*box_slice)(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs);
} GuidedContext;

#define OFFSET(x) offsetof(GuidedContext, x)
#define TFLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_RUNTIME_PARAM
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM

static const AVOption guided_options[] = {
    { "radius",   "set the box radius",                               OFFSET(radius),   AV_OPT_TYPE_INT,   {.i64 = 3    },     1,                    20, TFLAGS },
    { "eps",      "set the regularization parameter (with square)",   OFFSET(eps),      AV_OPT_TYPE_FLOAT, {.dbl = 0.01 },   0.0,                     1, TFLAGS },
    { "mode",     "set filtering mode (0: basic mode; 1: fast mode)", OFFSET(mode),     AV_OPT_TYPE_INT,   {.i64 = BASIC}, BASIC,          NB_MODES - 1, TFLAGS, .unit = "mode" },
    { "basic",    "basic guided filter",                              0,                AV_OPT_TYPE_CONST, {.i64 = BASIC},     0,                     0, TFLAGS, .unit = "mode" },
    { "fast",     "fast guided filter",                               0,                AV_OPT_TYPE_CONST, {.i64 = FAST },     0,                     0, TFLAGS, .unit = "mode" },
    { "sub",      "subsampling ratio for fast mode",                  OFFSET(sub),      AV_OPT_TYPE_INT,   {.i64 = 4    },     2,                    64, TFLAGS },
    { "guidance", "set guidance mode (0: off mode; 1: on mode)",      OFFSET(guidance), AV_OPT_TYPE_INT,   {.i64 = OFF  },   OFF, NB_GUIDANCE_MODES - 1,  FLAGS, .unit = "guidance" },
    { "off",      "only one input is enabled",                        0,                AV_OPT_TYPE_CONST, {.i64 = OFF  },     0,                     0,  FLAGS, .unit = "guidance" },
    { "on",       "two inputs are required",                          0,                AV_OPT_TYPE_CONST, {.i64 = ON   },     0,                     0,  FLAGS, .unit = "guidance" },
    { "planes",   "set planes to filter",                             OFFSET(planes),   AV_OPT_TYPE_INT,   {.i64 = 1    },     0,                   0xF, TFLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(guided);

typedef struct ThreadData {
  size_t n;     
  size_t m;     
  size_t ld_n;   
  size_t ld_m;  
  float *src;   
  float *work;   
  float *ai;
  float *bi;  
} ThreadData;


#define SWAP(a, b)				\
  do {						\
    __auto_type _t = (a);				\
    (a) = (b);					\
    (b) = (_t);					\
  } while(false)


#define V 8
// log2(V)
#define POW_V  3

//number of registers to use in a loop
#define NR 16   

// log2(V*NR)
#define POW_T  7  

/*
  block size for transpose.  If value is changed,
  correct also POW_BL and BL_V below!!
*/

#define BL 64
// log2(BL)
#define POW_BL 6
//BL divided by V
#define BL_V 8
#define BL_V_HALF 4

typedef __m256 float_packed;

#define MUL(a,b)     _mm256_mul_ps(a,b)
#define DIV(a,b)     _mm256_div_ps(a,b)
#define ADD(a,b)     _mm256_add_ps(a,b)
#define SUB(a,b)     _mm256_sub_ps(a,b)
#define LOAD(a)      _mm256_load_ps(&a)
#define STORE(a,b)   _mm256_store_ps(&a,b)
#define BROADCAST(a) _mm256_broadcast_ss(&a)


void boxfilter1D(const float *x_in, float *x_out, size_t r, size_t n, size_t m, size_t ld);

void boxfilter1D_norm(const float *x_in, float *x_out, size_t r, size_t n, size_t m, size_t ld, const float * a_norm, const float * b_norm);


void transpose_8x8_2(float * a, float * b, size_t n, size_t m);
  
void transpose_8x8(float * a, float * b, size_t n, size_t m);

void transpose(float * in, float * out, size_t ld_n, size_t ld_m);

void matmul(const float *x1, const float *x2, float *y, size_t ld_n, size_t m);

void diffmatmul(float *x1, const float *x2, const float * x3, size_t ld_n, size_t m);

void addmatmul(float *x1, const float *x2, const float * x3, size_t ld_n, size_t m);

void matdivconst(float *x1, const float *x2, size_t ld_n, size_t m, float e);




alignas(32) const float zero = 0;


static int boxfilter(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs)
{
  GuidedContext *s = ctx->priv;
  ThreadData *t = arg;
  
  const size_t n = t->n;
  const size_t m = t->m;
  const size_t ld_n = t->ld_n;
  const size_t ld_m = t->ld_m;
  //  const int slice_start = (height * jobnr) / nb_jobs;
  //  const int slice_end   = (height * (jobnr + 1)) / nb_jobs;
  const size_t r = s->radius;
  float *x    = t->src;
  float *work = t->work;
  float *ai   = t->ai;
  float *bi    = t->bi;
  
  boxfilter1D(x, work, r, n, m, ld_n);
  
  transpose(work, x, ld_n, ld_m); 
  
  boxfilter1D_norm(x, work, r, m, n, ld_m, ai, bi);
  
  /* transpose(work, x, ld_m, ld_n); */
  
  return 0;
}

static const enum AVPixelFormat pix_fmts[] = {
  AV_PIX_FMT_YUVA444P, AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV440P,
  AV_PIX_FMT_YUVJ444P, AV_PIX_FMT_YUVJ440P,
  AV_PIX_FMT_YUVA422P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUVA420P, AV_PIX_FMT_YUV420P,
  AV_PIX_FMT_YUVJ422P, AV_PIX_FMT_YUVJ420P,
  AV_PIX_FMT_YUVJ411P, AV_PIX_FMT_YUV411P, AV_PIX_FMT_YUV410P,
  AV_PIX_FMT_YUV420P9, AV_PIX_FMT_YUV422P9, AV_PIX_FMT_YUV444P9,
  AV_PIX_FMT_YUV420P10, AV_PIX_FMT_YUV422P10, AV_PIX_FMT_YUV444P10,
  AV_PIX_FMT_YUV420P12, AV_PIX_FMT_YUV422P12, AV_PIX_FMT_YUV444P12, AV_PIX_FMT_YUV440P12,
  AV_PIX_FMT_YUV420P14, AV_PIX_FMT_YUV422P14, AV_PIX_FMT_YUV444P14,
  AV_PIX_FMT_YUV420P16, AV_PIX_FMT_YUV422P16, AV_PIX_FMT_YUV444P16,
  AV_PIX_FMT_YUVA420P9, AV_PIX_FMT_YUVA422P9, AV_PIX_FMT_YUVA444P9,
  AV_PIX_FMT_YUVA420P10, AV_PIX_FMT_YUVA422P10, AV_PIX_FMT_YUVA444P10,
  AV_PIX_FMT_YUVA420P16, AV_PIX_FMT_YUVA422P16, AV_PIX_FMT_YUVA444P16,
  AV_PIX_FMT_GBRP, AV_PIX_FMT_GBRP9, AV_PIX_FMT_GBRP10,
  AV_PIX_FMT_GBRP12, AV_PIX_FMT_GBRP14, AV_PIX_FMT_GBRP16,
  AV_PIX_FMT_GBRAP, AV_PIX_FMT_GBRAP10, AV_PIX_FMT_GBRAP12, AV_PIX_FMT_GBRAP16,
  AV_PIX_FMT_GRAY8, AV_PIX_FMT_GRAY9, AV_PIX_FMT_GRAY10, AV_PIX_FMT_GRAY12, AV_PIX_FMT_GRAY14, AV_PIX_FMT_GRAY16,
  AV_PIX_FMT_NONE
};

static int config_input(AVFilterLink *inlink)
{
  AVFilterContext *ctx = inlink->dst;
  GuidedContext *s = ctx->priv;
  const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
  
  if (s->mode == BASIC) {
    s->sub = 1;
  } else if (s->mode == FAST) {
    if (s->radius >= s->sub)
      s->radius = s->radius / s->sub;
    else {
      s->radius = 1;
    }
  }
  
  s->depth = desc->comp[0].depth;
  s->width = ctx->inputs[0]->w;
  s->height = ctx->inputs[0]->h;
  
  s->planewidth[1]  = s->planewidth[2] = AV_CEIL_RSHIFT(inlink->w, desc->log2_chroma_w);
  s->planewidth[0]  = s->planewidth[3] = inlink->w;
  s->planeheight[1] = s->planeheight[2] = AV_CEIL_RSHIFT(inlink->h, desc->log2_chroma_h);
  s->planeheight[0] = s->planeheight[3] = inlink->h;
  
  s->nb_planes = av_pix_fmt_count_planes(inlink->format);
  s->box_slice = boxfilter;
  return 0;
}



static int guided_byte(AVFilterContext *ctx, GuidedContext *s,                        
		       const uint8_t *ssrc, const uint8_t *ssrcRef,                  
		       uint8_t *ddst, int radius, float eps, int width, int height,  
		       int src_stride, int src_ref_stride, int dst_stride,           
		       float maxval)                                                 
{                                                                                       
  int ret = 0;                                                                        
  uint8_t *dst = (uint8_t *)ddst;                                                           
  const uint8_t *src = (const uint8_t *)ssrc;                                               
  const uint8_t *srcRef = (const uint8_t *)ssrcRef;                                         
  
  /* int sub = s->sub;                                                                    */
  /* int h = (height % sub) == 0 ? height / sub : height / sub + 1;                       */
  /* int w = (width % sub) == 0 ? width / sub : width / sub + 1;              */            
  

  size_t n = width;
  size_t m = height;
  size_t r = radius;
  
  size_t ld_n, ld_m, i, j;
  
  ThreadData t;
  
  
  const int nb_threads = ff_filter_get_nb_threads(ctx);

 

  float *I = s->I;
  
  float *p = s->p;
  
  float *work = s->work;
  
  float *t1 = s->t1;
  float *t2 = s->t2;
  float *t3 = s->t3;
  
  float * ai = s->ai;
  float * bi = s->bi;

  float * ai2 = s->ai2;

  float maxval_reciprocal = 1/maxval;
  

  ld_n = (n>>POW_V)<<POW_V;
  
  if (ld_n < n) ld_n = ld_n + V;
  
  ld_m = (m>>POW_V)<<POW_V;
  
  if (ld_m < m) ld_m = ld_m + V;
  
  if (n < 2 * r + 1 || m < 2 * r + 1 )
    {
      return -1;
    }
  
  //  printf("nb_threads = %d, n = %lu, m = %lu, ld_n = %lu, ld_m = %lu, ssrc = %p, ssrcRef = %p, src_stride = %d, src_ref_stride = %d, dst_stride = %d\n", nb_threads, n, m, ld_n, ld_m, ssrc, ssrcRef, src_stride, src_ref_stride, dst_stride);
    
  for (i = 0; i < r+1; i++)
    {
      ai[i] = 1./(r+1+i)/(2*r+1);
      bi[i] = 1./(r+1+i)*(2*r+1);
    }
  
  for (i = r+1; i < m - (r+1); i++)
    {
      ai[i] = 1./(2*r+1)/(2*r+1);
    }
  
  for (i = m - (r+1); i < m; i++)
    {
      ai[i] = ai[m - 1 - i];
    }
  
  for (i = m; i < ld_m; i++)
    {
      ai[i] = 0;
    }

  
  
  for (i = 0; i < r+1; i++)
    {
      ai2[i] = 1./(r+1+i)/(2*r+1);
    }
  
  for (i = r+1; i < n - (r+1); i++)
    {
      ai2[i] = 1./(2*r+1)/(2*r+1);
    }
  
  for (i = n - (r+1); i < n; i++)
    {
      ai2[i] = ai[n - 1 - i];
    }

  for (i = n; i < ld_n; i++)
    {
      ai2[i] = 0;
    }

 

  
  for (j = 0; j<m; j++)
    {
      for (i = 0; i<n; i++)
	{
	  I[j * ld_n + i] = (float) src[j * src_stride + i] * maxval_reciprocal;
	}
    }
  
  memcpy(t1, I, ld_n * ld_m * sizeof(float));
  if (ssrcRef == ssrc && src_stride == src_ref_stride)
    {
      memcpy(p, I,  ld_n * ld_m * sizeof(float));
    }
  else
    {
      for (j = 0; j<m; j++)
	{
	  for (i = 0; i<n; i++)
	    {
	      p[j * ld_n + i] = (float) srcRef[j * src_ref_stride + i] * maxval_reciprocal;
	    }
	}
    }
    
  t.n    = n;                                                                       
  t.m    = m;                                                                       
  t.ld_n = ld_n;                                                                    
  t.ld_m = ld_m;
  t.work = work;
  t.ai   = ai;
  t.bi   = bi;
   
  
  // boxfilter(t1, r, n, m, ld_n, ld_m, ai, bi, work); //t1 = mean_I  (m x n)
  t.src = t1; t.work = work; ff_filter_execute(ctx, s->box_slice, &t, NULL, FFMIN(m, nb_threads));  SWAP(t1, work);
  
  matmul(I, I, t2, ld_n, m);//t2 = I*I (n x m)
  
  // boxfilter(t2, r, n, m, ld_n, ld_m, ai, bi, work);//t2 = mean_II  (m x n)
  t.src = t2; t.work = work; ff_filter_execute(ctx, s->box_slice, &t, NULL, FFMIN(m, nb_threads));  SWAP(t2, work);
  
  diffmatmul(t2, t1, t1, ld_n, m);//t2 = mean_II - mean_I * mean_I = var_I  (m x n)
  
  matmul(I, p, t3, ld_n, m);//t3 = I*p   (n x m)
  
  //boxfilter(t3, r, n, m, ld_n, ld_m, ai, bi, work);//t3 = mean_Ip  (m x n)
  t.src = t3; t.work = work; ff_filter_execute(ctx, s->box_slice, &t, NULL, FFMIN(m, nb_threads));  SWAP(t3, work);
  
  //boxfilter(p, r, n, m, ld_n, ld_n, ai, bi, work);//p = mean_p;   (m x n)
  t.src = p; t.work = work; ff_filter_execute(ctx, s->box_slice, &t, NULL, FFMIN(m, nb_threads)); SWAP(p, work);
  
  diffmatmul(t3, t1, p, ld_n, m);//t3 = mean_Ip - mean_I * mean_p = cov_Ip  (m x n)
  
  matdivconst(t3, t2, ld_n, m, eps);//t3 = t3/(t2+eps) = cov_Ip/(var_I + eps) = a  (m x n)
  
  diffmatmul(p, t3, t1, ld_n, m);//p = mean_p - a * mean_I = b  (m x n)

  //For the next two boxfilters, we need to apply them on transposed images
  t.n    = m;                                                                       
  t.m    = n;                                                                       
  t.ld_n = ld_m;                                                                    
  t.ld_m = ld_n;
  t.work = work;
  t.ai   = ai2;
  t.bi   = bi;
  
  //boxfilter(t3, r, n, m, ld_n, ld_m, ai, bi, work);//t3 = mean_a  (n x m)
  t.src = t3; t.work = work; ff_filter_execute(ctx, s->box_slice, &t, NULL, FFMIN(m, nb_threads)); SWAP(t3, work);
  
  //boxfilter(p,  r, n, m, ld_n, ld_m, ai, bi, work);//p = mean_b  (n x m)
  t.src = p;  t.work = work; ff_filter_execute(ctx, s->box_slice, &t, NULL, FFMIN(m, nb_threads)); SWAP(p, work);

  addmatmul(p, t3, I, ld_n, m);//p = mean_b + mean_a * I = q
 
  for (j = 0; j<m; j++)
    {   
      for (i = 0; i<n; i++)
	{
	  dst [j * dst_stride + i] = (uint8_t) (p[j * ld_n + i] * maxval);
	}
    }
                                                                                        
                                                                                        
  return ret;                                                                         
}

  
static int filter_frame(AVFilterContext *ctx, AVFrame **out, AVFrame *in, AVFrame *ref)
{
    GuidedContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    *out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!*out)
        return AVERROR(ENOMEM);
    av_frame_copy_props(*out, in);

    for (int plane = 0; plane < s->nb_planes; plane++) {
        if (!(s->planes & (1 << plane))) {
            av_image_copy_plane((*out)->data[plane], (*out)->linesize[plane],
                                in->data[plane], in->linesize[plane],
                                s->planewidth[plane] * ((s->depth + 7) / 8), s->planeheight[plane]);
            continue;
        }
        if (s->depth <= 8)
            guided_byte(ctx, s, in->data[plane], ref->data[plane], (*out)->data[plane], s->radius, s->eps,
                        s->planewidth[plane], s->planeheight[plane],
                        in->linesize[plane], ref->linesize[plane], (*out)->linesize[plane], (1 << s->depth) - 1.f);
	/*        else */
	  
	  /* guided_word(ctx, s, in->data[plane], ref->data[plane], (*out)->data[plane], s->radius, s->eps, */
          /*               s->planewidth[plane], s->planeheight[plane], */
          /*               in->linesize[plane] / 2, ref->linesize[plane] / 2, (*out)->linesize[plane] / 2, (1 << s->depth) - 1.f); */
    }

    return 0;
}

static int process_frame(FFFrameSync *fs)
{
    AVFilterContext *ctx = fs->parent;
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame *out_frame = NULL, *main_frame = NULL, *ref_frame = NULL;
    int ret;
    ret = ff_framesync_dualinput_get(fs, &main_frame, &ref_frame);
    if (ret < 0)
        return ret;

    if (ctx->is_disabled)
        return ff_filter_frame(outlink, main_frame);

    ret = filter_frame(ctx, &out_frame, main_frame, ref_frame);
    if (ret < 0)
        return ret;
    av_frame_free(&main_frame);

    return ff_filter_frame(outlink, out_frame);
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    GuidedContext *s = ctx->priv;
    AVFilterLink *mainlink = ctx->inputs[0];
    FilterLink         *il = ff_filter_link(mainlink);
    FilterLink         *ol = ff_filter_link(outlink);
    FFFrameSyncIn *in;
    int w, h, r, ret;

    size_t n, m, ld_n, ld_m;

    r = s->radius;

    if (s->guidance == ON) {
        if (ctx->inputs[0]->w != ctx->inputs[1]->w ||
            ctx->inputs[0]->h != ctx->inputs[1]->h) {
            av_log(ctx, AV_LOG_ERROR, "Width and height of input videos must be same.\n");
            return AVERROR(EINVAL);
        }
    }

    outlink->w = w = mainlink->w;
    outlink->h = h = mainlink->h;
    outlink->time_base = mainlink->time_base;
    outlink->sample_aspect_ratio = mainlink->sample_aspect_ratio;
    ol->frame_rate = il->frame_rate;

    n = w;
    m = h;
    
    ld_n = (n>>POW_V)<<POW_V;
    
    if (ld_n < n) ld_n = ld_n + V;
    
    ld_m = (m>>POW_V)<<POW_V;
    
    if (ld_m < m) ld_m = ld_m + V;

    s->I      = av_calloc(ld_n * ld_m, sizeof(*s->I));
    s->p      = av_calloc(ld_n * ld_m, sizeof(*s->p));

    s->work   = av_calloc(ld_n * ld_m, sizeof(*s->work));
    s->t1     = av_calloc(ld_n * ld_m, sizeof(*s->t1));
    s->t2     = av_calloc(ld_n * ld_m, sizeof(*s->t2));
    s->t3     = av_calloc(ld_n * ld_m, sizeof(*s->t3));

    s->ai     = av_calloc(ld_m, sizeof(*s->ai));
    s->bi     = av_calloc(r+1,  sizeof(*s->bi));

    s->ai2     = av_calloc(ld_n, sizeof(*s->ai2));
    
    

    if (!s->I || !s->p || !s->work || !s->t1 || !s->t2 || !s->t3 || !s->ai || !s->bi || !s->ai2)
        return AVERROR(ENOMEM);

    if (s->guidance == OFF)
        return 0;

    if ((ret = ff_framesync_init(&s->fs, ctx, 2)) < 0)
        return ret;

    outlink->time_base = s->fs.time_base;

    in = s->fs.in;
    in[0].time_base = mainlink->time_base;
    in[1].time_base = ctx->inputs[1]->time_base;
    in[0].sync   = 2;
    in[0].before = EXT_INFINITY;
    in[0].after  = EXT_INFINITY;
    in[1].sync   = 1;
    in[1].before = EXT_INFINITY;
    in[1].after  = EXT_INFINITY;
    s->fs.opaque   = s;
    s->fs.on_event = process_frame;

    return ff_framesync_configure(&s->fs);
}

static int activate(AVFilterContext *ctx)
{
    GuidedContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    AVFilterLink *inlink = ctx->inputs[0];
    AVFrame *frame = NULL;
    AVFrame *out = NULL;
    int ret, status;
    int64_t pts;
    if (s->guidance)
        return ff_framesync_activate(&s->fs);

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    if ((ret = ff_inlink_consume_frame(inlink, &frame)) > 0) {
        if (ctx->is_disabled)
            return ff_filter_frame(outlink, frame);

        ret = filter_frame(ctx, &out, frame, frame);
        av_frame_free(&frame);
        if (ret < 0)
            return ret;
        ret = ff_filter_frame(outlink, out);
    }
    if (ret < 0)
        return ret;
    if (ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        ff_outlink_set_status(outlink, status, pts);
        return 0;
    }
    if (ff_outlink_frame_wanted(outlink))
        ff_inlink_request_frame(inlink);
    return 0;
}

static av_cold int init(AVFilterContext *ctx)
{
    GuidedContext *s = ctx->priv;
    AVFilterPad pad = { 0 };
    int ret;

    pad.type         = AVMEDIA_TYPE_VIDEO;
    pad.name         = "source";
    pad.config_props = config_input;

    if ((ret = ff_append_inpad(ctx, &pad)) < 0)
        return ret;

    if (s->guidance == ON) {
        pad.type         = AVMEDIA_TYPE_VIDEO;
        pad.name         = "guidance";
        pad.config_props = NULL;

        if ((ret = ff_append_inpad(ctx, &pad)) < 0)
            return ret;
    }

    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    GuidedContext *s = ctx->priv;
    if (s->guidance == ON)
        ff_framesync_uninit(&s->fs);

    av_freep(&s->I);
    av_freep(&s->p);
    av_freep(&s->work);
    av_freep(&s->t1);
    av_freep(&s->t2);
    av_freep(&s->t3);
    av_freep(&s->ai);
    av_freep(&s->bi);

    av_freep(&s->ai2);
    
    return;
}

static const AVFilterPad guided_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
    },
};

const AVFilter ff_vf_guided = {
    .name            = "guided",
    .description     = NULL_IF_CONFIG_SMALL("Apply Guided filter."),
    .init            = init,
    .uninit          = uninit,
    .priv_size       = sizeof(GuidedContext),
    .priv_class      = &guided_class,
    .activate        = activate,
    .inputs          = NULL,
    FILTER_OUTPUTS(guided_outputs),
    FILTER_PIXFMTS_ARRAY(pix_fmts),
    .flags           = AVFILTER_FLAG_DYNAMIC_INPUTS | AVFILTER_FLAG_SLICE_THREADS |
                       AVFILTER_FLAG_SUPPORT_TIMELINE_INTERNAL,
    .process_command = ff_filter_process_command,
};

void transpose_8x8_2(float * a, float * b, size_t n, size_t m)
{
  float_packed v0, v1, v2, v3, v4, v5, v6, v7;
  float_packed s0, s1, s2, s3, s4, s5, s6, s7;

  float_packed w0, w1, w2, w3, w4, w5, w6, w7;
  float_packed t0, t1, t2, t3, t4, t5, t6, t7;
  

  v0 = LOAD(a[0]);
  v1 = LOAD(a[n]);
  v2 = LOAD(a[2*n]);
  v3 = LOAD(a[3*n]);
  v4 = LOAD(a[4*n]);
  v5 = LOAD(a[5*n]);
  v6 = LOAD(a[6*n]);
  v7 = LOAD(a[7*n]);

  a += V;

  w0 = LOAD(a[0]);
  w1 = LOAD(a[n]);
  w2 = LOAD(a[2*n]);
  w3 = LOAD(a[3*n]);
  w4 = LOAD(a[4*n]);
  w5 = LOAD(a[5*n]);
  w6 = LOAD(a[6*n]);
  w7 = LOAD(a[7*n]);
  
  
  s0 = _mm256_unpacklo_ps(v0, v1);
  s1 = _mm256_unpackhi_ps(v0, v1);
  s2 = _mm256_unpacklo_ps(v2, v3);
  s3 = _mm256_unpackhi_ps(v2, v3);
  s4 = _mm256_unpacklo_ps(v4, v5);
  s5 = _mm256_unpackhi_ps(v4, v5);
  s6 = _mm256_unpacklo_ps(v6, v7);
  s7 = _mm256_unpackhi_ps(v6, v7);
  
  v0 = _mm256_shuffle_ps(s0,s2,_MM_SHUFFLE(1,0,1,0));  
  v1 = _mm256_shuffle_ps(s0,s2,_MM_SHUFFLE(3,2,3,2));
  v2 = _mm256_shuffle_ps(s1,s3,_MM_SHUFFLE(1,0,1,0));
  v3 = _mm256_shuffle_ps(s1,s3,_MM_SHUFFLE(3,2,3,2));
  v4 = _mm256_shuffle_ps(s4,s6,_MM_SHUFFLE(1,0,1,0));
  v5 = _mm256_shuffle_ps(s4,s6,_MM_SHUFFLE(3,2,3,2));
  v6 = _mm256_shuffle_ps(s5,s7,_MM_SHUFFLE(1,0,1,0));
  v7 = _mm256_shuffle_ps(s5,s7,_MM_SHUFFLE(3,2,3,2));
  
  s0 = _mm256_permute2f128_ps(v0, v4, 0x20);
  s1 = _mm256_permute2f128_ps(v1, v5, 0x20);
  s2 = _mm256_permute2f128_ps(v2, v6, 0x20);
  s3 = _mm256_permute2f128_ps(v3, v7, 0x20);
  s4 = _mm256_permute2f128_ps(v0, v4, 0x31);
  s5 = _mm256_permute2f128_ps(v1, v5, 0x31);
  s6 = _mm256_permute2f128_ps(v2, v6, 0x31);
  s7 = _mm256_permute2f128_ps(v3, v7, 0x31);

  t0 = _mm256_unpacklo_ps(w0, w1);
  t1 = _mm256_unpackhi_ps(w0, w1);
  t2 = _mm256_unpacklo_ps(w2, w3);
  t3 = _mm256_unpackhi_ps(w2, w3);
  t4 = _mm256_unpacklo_ps(w4, w5);
  t5 = _mm256_unpackhi_ps(w4, w5);
  t6 = _mm256_unpacklo_ps(w6, w7);
  t7 = _mm256_unpackhi_ps(w6, w7);
  
  w0 = _mm256_shuffle_ps(t0,t2,_MM_SHUFFLE(1,0,1,0));  
  w1 = _mm256_shuffle_ps(t0,t2,_MM_SHUFFLE(3,2,3,2));
  w2 = _mm256_shuffle_ps(t1,t3,_MM_SHUFFLE(1,0,1,0));
  w3 = _mm256_shuffle_ps(t1,t3,_MM_SHUFFLE(3,2,3,2));
  w4 = _mm256_shuffle_ps(t4,t6,_MM_SHUFFLE(1,0,1,0));
  w5 = _mm256_shuffle_ps(t4,t6,_MM_SHUFFLE(3,2,3,2));
  w6 = _mm256_shuffle_ps(t5,t7,_MM_SHUFFLE(1,0,1,0));
  w7 = _mm256_shuffle_ps(t5,t7,_MM_SHUFFLE(3,2,3,2));
  
  t0 = _mm256_permute2f128_ps(w0, w4, 0x20);
  t1 = _mm256_permute2f128_ps(w1, w5, 0x20);
  t2 = _mm256_permute2f128_ps(w2, w6, 0x20);
  t3 = _mm256_permute2f128_ps(w3, w7, 0x20);
  t4 = _mm256_permute2f128_ps(w0, w4, 0x31);
  t5 = _mm256_permute2f128_ps(w1, w5, 0x31);
  t6 = _mm256_permute2f128_ps(w2, w6, 0x31);
  t7 = _mm256_permute2f128_ps(w3, w7, 0x31);
  
  STORE(b[0],s0);
  STORE(b[m],s1);
  STORE(b[2*m],s2);
  STORE(b[3*m],s3);
  STORE(b[4*m],s4);
  STORE(b[5*m],s5);
  STORE(b[6*m],s6);
  STORE(b[7*m],s7);

  b += m << POW_V;  //b = b + m * V

  STORE(b[0],t0);
  STORE(b[m],t1);
  STORE(b[2*m],t2);
  STORE(b[3*m],t3);
  STORE(b[4*m],t4);
  STORE(b[5*m],t5);
  STORE(b[6*m],t6);
  STORE(b[7*m],t7);

  
}


void transpose_8x8(float * a, float * b, size_t n, size_t m)
{
  float_packed v0, v1, v2, v3, v4, v5, v6, v7;
  float_packed s0, s1, s2, s3, s4, s5, s6, s7;

  
  v0 = LOAD(a[0]);				
  v1 = LOAD(a[n]);				
  v2 = LOAD(a[2*n]);				
  v3 = LOAD(a[3*n]);				
  v4 = LOAD(a[4*n]);				
  v5 = LOAD(a[5*n]);				
  v6 = LOAD(a[6*n]);				
  v7 = LOAD(a[7*n]);				
  s0 = _mm256_unpacklo_ps(v0, v1);		
  s1 = _mm256_unpackhi_ps(v0, v1);		
  s2 = _mm256_unpacklo_ps(v2, v3);		
  s3 = _mm256_unpackhi_ps(v2, v3);		
  s4 = _mm256_unpacklo_ps(v4, v5);		
  s5 = _mm256_unpackhi_ps(v4, v5);		
  s6 = _mm256_unpacklo_ps(v6, v7);		
  s7 = _mm256_unpackhi_ps(v6, v7);			
  v0 = _mm256_shuffle_ps(s0,s2,_MM_SHUFFLE(1,0,1,0));	
  v1 = _mm256_shuffle_ps(s0,s2,_MM_SHUFFLE(3,2,3,2));	
  v2 = _mm256_shuffle_ps(s1,s3,_MM_SHUFFLE(1,0,1,0));	
  v3 = _mm256_shuffle_ps(s1,s3,_MM_SHUFFLE(3,2,3,2));	
  v4 = _mm256_shuffle_ps(s4,s6,_MM_SHUFFLE(1,0,1,0));	
  v5 = _mm256_shuffle_ps(s4,s6,_MM_SHUFFLE(3,2,3,2));	
  v6 = _mm256_shuffle_ps(s5,s7,_MM_SHUFFLE(1,0,1,0));	
  v7 = _mm256_shuffle_ps(s5,s7,_MM_SHUFFLE(3,2,3,2));	
  s0 = _mm256_permute2f128_ps(v0, v4, 0x20);		
  s1 = _mm256_permute2f128_ps(v1, v5, 0x20);		
  s2 = _mm256_permute2f128_ps(v2, v6, 0x20);		
  s3 = _mm256_permute2f128_ps(v3, v7, 0x20);		
  s4 = _mm256_permute2f128_ps(v0, v4, 0x31);		
  s5 = _mm256_permute2f128_ps(v1, v5, 0x31);		
  s6 = _mm256_permute2f128_ps(v2, v6, 0x31);		
  s7 = _mm256_permute2f128_ps(v3, v7, 0x31);		
  STORE(b[0],s0);					
  STORE(b[m],s1);					
  STORE(b[2*m],s2);					
  STORE(b[3*m],s3);					
  STORE(b[4*m],s4);					
  STORE(b[5*m],s5);					
  STORE(b[6*m],s6);					
  STORE(b[7*m],s7);					
}

void transpose(float * in, float * out, size_t n, size_t m)
{
  /*
    Works only for n and m both divisible by V
  */

  
  size_t ni = n>>POW_BL;

  size_t mi = m>>POW_BL;

  size_t rest_n, rest_m;
  


  float *a0 = in;
  float *b0 = out;

  float *a;
  float *b;

  size_t k, l, i, j;
  
  rest_n = n - (ni << POW_BL);
  rest_n = rest_n >> POW_V;
  
  rest_m = m - (mi << POW_BL);
  rest_m = rest_m >> POW_V;

  //printf("rest_n = %ld\n", rest_n);
  //printf("rest_m = %ld\n", rest_m);

  for (k = 0; k < mi; k++)
    {
      for (l = 0; l < ni; l++)
      {
	a = a0;
	b = b0;
	for (i = 0; i<BL_V; i++)
	  {
	    for (j = 0; j<BL_V_HALF; j++)
	    {
	      transpose_8x8_2(a, b, n, m);
       	      a += V<<1;
	      b += (m << POW_V)*2;  //b = b + m * V
	    }
	    a -= BL;            //a = a - BL_V * V = a - BL
	    a += n << POW_V;    //a = a + n * V
            b -= m << POW_BL;   //b = b - m * BL_V * V = b - m * BL
	    b += V;
	  }
	    
	a0 += BL;
	b0 += m << POW_BL;
      }
      if (rest_n != 0)
	{
	  a = a0;
	  b = b0;
	  for (i = 0; i<BL_V; i++)
	    {
	      for (j = 0; j<rest_n; j++)
		{
		  transpose_8x8(a, b, n, m);
		  a += V;
		  b += m << POW_V;
		}
	      a -= rest_n << POW_V;
	      a += n << POW_V;
	      b -= (rest_n * m) << POW_V;
	      b += V;
	    }
	}

      a0 -= ni << POW_BL;
      a0 += n <<POW_BL;

      b0 -= (ni * m) << POW_BL;
      b0 += BL;
    }

  if (rest_m != 0)
    {
      for (l = 0; l < ni; l++)
      {
	a = a0;
	b = b0;
	for (i = 0; i<rest_m; i++)
	  {
	    for (j = 0; j<BL_V; j++)
	    {
	      transpose_8x8(a, b, n, m);
       	      a += V;
	      b += m << POW_V;  //b = b + m * V
	    }
	    a -= BL;            //a = a - BL_V * V = a - BL
	    a += n << POW_V;    //a = a + n * V
            b -= m << POW_BL;   //b = b - m * BL_V * V = b - m * BL
	    b += V;
	  }
	    
	a0 += BL;
	b0 += m << POW_BL;
      }
      if (rest_n != 0)
	{
	  a = a0;
	  b = b0;
	  for (i = 0; i<rest_m; i++)
	    {
	      for (j = 0; j<rest_n; j++)
		{
		  transpose_8x8(a, b, n, m);
		  a += V;
		  b += m << POW_V;
		}
	      a -= rest_n << POW_V;
	      a += n << POW_V;
	      b -= (rest_n * m) << POW_V;
	      b += V;
	    }
	}
    }

}

void matmul(const float *x1, const float *x2, float *y, size_t ld_n, size_t m)
{
  //y = x1.*x2

  size_t i, ld_red;

  float_packed v1, v2, vy;

  const float *a1 = x1;
  const float *a2 = x2;
  float *b  = y;

  ld_red = ld_n >> POW_V;

  i = m * ld_red;
  do
    {
      v1 = LOAD(a1[0]);
      v2 = LOAD(a2[0]);
      vy = MUL(v1, v2);
      STORE(b[0], vy);
      a1 += V;
      a2 += V;
      b  += V;
    } while (i -= 1);
}



void diffmatmul(float *x1, const float *x2, const float * x3, size_t ld_n, size_t m)
{

  //x1 = x1 - x2 * x3

  size_t i, ld_red;

  float_packed v1, v2, v3, vy;

  float *a1 = x1;
  const float *a2 = x2;
  const float *a3 = x3;
  
  ld_red = ld_n >> POW_V;


  i = m * ld_red;
  do
    {

      v1 = LOAD(a1[0]);
      v2 = LOAD(a2[0]);
      v3 = LOAD(a3[0]);
      vy = MUL(v2, v3);
      vy = SUB(v1, vy);
      STORE(a1[0], vy);
      a1 += V;
      a2 += V;
      a3 += V;
    } while (i -= 1);
}


void addmatmul(float *x1, const float *x2, const float * x3, size_t ld_n, size_t m)
{

  //x1 = x1 + x2 * x3
  size_t i, ld_red;

  float_packed v1, v2, v3, vy;

  float *a1 = x1;
  const float *a2 = x2;
  const float *a3 = x3;
  
  ld_red = ld_n >> POW_V;

  i = m * ld_red;
  do
    {
      v1 = LOAD(a1[0]);
      v2 = LOAD(a2[0]);
      v3 = LOAD(a3[0]);
      vy = MUL(v2, v3);
      vy = ADD(v1, vy);
      STORE(a1[0], vy);
      a1 += V;
      a2 += V;
      a3 += V;
    } while (i -= 1);
}



void matdivconst(float *x1, const float *x2,  size_t ld_n, size_t m, float e)
{
  //x1 = x1./(x2+eps)
  
  size_t i, ld_red;

  float_packed v1, v2, vy, ve;

  float *a1 = x1;
  const float *a2 = x2;

  ve = BROADCAST(e);
  
  ld_red = ld_n >> POW_V;
  
  i = m * ld_red;
  do
    {
      v1 = LOAD(a1[0]);
      v2 = LOAD(a2[0]);
      v2 = ADD(v2, ve);
      vy = DIV(v1, v2);
      STORE(a1[0], vy);
      a1 += V;
      a2 += V;
    } while (i -= 1);
}


void boxfilter1D(const float *restrict x_in, float *restrict x_out, size_t r, size_t n, size_t m, size_t ld)
{
  
  /*
    - x_in  = pointer to 2D image n x n as 1D array
    
    - x_out = pointer to output 2D image n x n as 1D array
            = result of filter applied in first (leading) dimension to x_in
    
    - r     = radius of filter, made of 2*r+1 ones
    
    - n,m   = size of image, both should be >= 2 * r + 1

    - ld    = leading dimension for x_in and x_out, should be divisible by V! 


    NOTICE 1: Both x_in and x_out should be aligned to 32bytes for
    avx2 and 64bytes for avx512.  This can be done for example using
    aligned_alloc instead of malloc:
    
    float *x_in = aligned_alloc(32, (n*n)*sizeof(float));

    or using alignas(32) float x_in[N*N] for static arrays. 
    
    NOTICE 2: If n is not divisible by V, consider a leading dimension
    ld divisible by V and save x_in as a submatrix of size n x n in a
    matrix ld x n (column-major) or n x ld (row-major).
    

    The function works for both column-major and row-major 2D arrays
    by using index arithmetic in a 1D array.
   
    
    Consider the row-major case.

    NR = number of registers, each with V float values (NR = 16, V = 8
    for avx2 with registers having 256 bits)
    
    Outer loop over NR*V columns
      
      Inner loop: sliding window over all lines

      Go to next NR*V columns (via a0 for x_in and b0 for x_out)

    End outer loop

    Repeat once the outer loop for the rest of columns up to next
    number divisible by V greater than n, therefore still using vector
    acceleration but less than NR registers. This works as leading
    dimension ld is divisible by 8!

    Example for n = 1005, leading dimension ld = 1008, NR = 16, V = 8, NR*V = 128

    Outer loop with NR registers and V values - 7 times = 896 lines

    Outer loop with 13+1 registers and V values - once = 112 lines (max (NR-1)*V lines)
    
    Total: 1008 lines. Note that the last 3 lines are present via ld but never used in x_in and x_out.



    The inner loop over columns is divided in 4 loops - first r columns, next r+1, central part, last r columns
 
    The pointers a and a_diff for x_in and b for x_out start with the first column, current lines (via a0 and b0)
   

    Inner central loop (with j = n - 2 * r - 1):
    
        s contains NR*V values saved already in x_out.
    
        Read NR*V values from the next line starting from pointer a
    
        Add them to s

        Read NR*V values from line - (2*r-1)

        Substract them from s

        Save s to x_out, starting from pointer b
    
        Point a and b go the next column, same lines, a = a + n, b = b + n

    End central loop


  */

      
  float_packed v[NR];
  float_packed s[NR];

  const float * a0 =  x_in;
  const float * a_diff = x_in;
  float * b0 = x_out;

  const float * a;
  float * b;
  
  size_t i, j, k, rest, ni;

  //Loop using NR registers with V values, with POW_T = log2(V*NR) 
  
  ni =  n >> POW_T; // ni = n/(V*NR);
  

  for (k = 0; k<ni; k++)
    {
      a = a0;
      b = b0;
      a_diff = a0;
     
      s[0] = BROADCAST(zero);
 
      for (i = 1; i<NR; i++)
	{
	  s[i] = s[0];
	}
      
      for (j = 0; j < r; j++)
	{
	  for (i = 0; i<NR; i++)
	    {
	      v[i] = LOAD(a[i*V]);
	      s[i] = ADD(s[i], v[i]);
	    }
          a += ld;	  
	} 
      
      for (j = 0; j < r + 1; j++)
	{
	  for (i = 0; i<NR; i++)
	    {
	      v[i] = LOAD(a[i*V]);
	      s[i] = ADD(s[i], v[i]);
	      STORE(b[i*V], s[i]);
	    }
          a += ld;
	  b += ld;
	} 

      for (j = 0; j < m - 2 * r - 1; j++)
	{
	  for (i = 0; i<NR; i++)
	    {
	      v[i] = LOAD(a[i*V]);
	      s[i] = ADD(s[i], v[i]);
	      v[i] = LOAD(a_diff[i*V]);
	      s[i] = SUB(s[i], v[i]);
	      STORE(b[i*V], s[i]);
	    }
          a += ld;
	  b += ld;
	  a_diff += ld;
	};

      for (j = 0; j < r; j++)
	{
	  for (i = 0; i<NR; i++)
	    {
	      v[i] = LOAD(a_diff[i*V]);
	      s[i] = SUB(s[i], v[i]);
	      STORE(b[i*V], s[i]);
	    }
	  b += ld;
	  a_diff += ld;
	  
	}

      a0 += NR*V;
      b0 += NR*V;	
            
    }

  
  rest  = n - (ni<<POW_T);

  if (rest == 0) return;

  ni = rest >> POW_V;  //ni = rest/V


  /* 
     This is for the rest of lines up to n (if not divisible by V*NR).
     
     This can be done using less than NR registers. 

     For arbitrary n, not divisible by V=8, there are still
     
     some values, less than V, to treat. In general, one would
     
     use mmx for 4 floats and then scalar to treat these last values.
     
     However, based on the leading dimension being divisible by V, 

     we can load the last V values in a vector register and do all
     
     calculations, without ever caring about the last ld-n values.
     
   */
  
  if (rest - (ni << POW_V) > 0)  ni = ni + 1;
  
  
  
  a = a0;
  b = b0;
  a_diff = a0;
  
  s[0] = BROADCAST(zero);
  for (i = 1; i<ni; i++)
    {
      s[i] = s[0];
    }

  for (j = 0; j < r; j++)
    {
      for (i = 0; i<ni; i++)
	{
	  v[i] = LOAD(a[i*V]);
	  s[i] = ADD(s[i], v[i]);
	    }
      a += ld;	  
    }
  
  for (j = 0; j < r + 1; j++)
    {
      for (i = 0; i<ni; i++)
	{
	  v[i] = LOAD(a[i*V]);
	  s[i] = ADD(s[i], v[i]);
	  STORE(b[i*V], s[i]);
	}
      a += ld;
      b += ld;
    } while (j -= 1);
  
  for (j = 0; j < m - 2 * r - 1; j++)
    {
      for (i = 0; i<ni; i++)
	{
	  v[i] = LOAD(a[i*V]);
	  s[i] = ADD(s[i], v[i]);
	  v[i] = LOAD(a_diff[i*V]);
	  s[i] = SUB(s[i], v[i]);
	  STORE(b[i*V], s[i]);
	}
      a += ld;
      b += ld;
      a_diff += ld;
    }
  
  for (j = 0; j < r; j++)
    {
      for (i = 0; i<ni; i++)
	{
	  v[i] = LOAD(a_diff[i*V]);
	  s[i] = SUB(s[i], v[i]);
	  STORE(b[i*V], s[i]);
	}
      b += ld;
      a_diff += ld;
      
    }
}


void boxfilter1D_norm(const float *restrict x_in, float *restrict x_out, size_t r, size_t n, size_t m, size_t ld, const float *restrict a_norm, const float * restrict b_norm)
{
  
  /*
    - x_in  = pointer to 2D image n x n as 1D array
    
    - x_out = pointer to output 2D image n x n as 1D array
            = result of filter applied in 1 dimension to x_in 
	      AND final normalization by multiplication with matrix 1./N 
    
    - r     = radius of filter, made of 2*r+1 ones
    
    - n,m   = size of image, both should be >= 2 * r + 1
    
    - ld    = leading dimension for x_in and x_out, should be divisible by V! 

    - a_norm = pointer to a vector of length ld
    
    - b_norm = pointer to a vector of length r+1

    Note: a_norm @ b_norm = 1./N, where @ is the tensor product and N
    is the normalization matrix.

    b_norm(k) = 1 for k > r && k < n - r 

    These values for b are not saved, as multiplication with 1 can be
    omitted!

    See further comments in boxfilter1D.c

  */

      
  float_packed v[NR];
  float_packed s[NR];
  float_packed va, vb;
 
  const float * a0 =  x_in;
  const float * a_diff = x_in;
  float * b0 = x_out;

  const float * a;
  float * b;

  const float * ai =  a_norm;
  
  size_t i, j, k, rest, ni;

  //Loop using NR registers with V values, with POW_T = log2(V*NR) 
  
  ni =  n >> POW_T; // ni = n/V/NR;
  
  //  k = ni;

  // printf("%ld\n", ni);
  
  for (k = 0; k<ni; k++)
    {
      a = a0;
      b = b0;
      a_diff = a0;
     
      s[0] = BROADCAST(zero);
 
      for (i = 1; i<NR; i++)
	{
	  s[i] = s[0];
	}
      
      for (j = 0; j < r; j++)
	{
	  for (i = 0; i<NR; i++)
	    {
	      v[i] = LOAD(a[i*V]);
	      s[i] = ADD(s[i], v[i]);
	    }
          a += ld;	  
	} 
      
      for (j = 0; j < r + 1; j++)
	{
	  vb = BROADCAST(b_norm[j]);
	  for (i = 0; i<NR; i++)
	    {
	      v[i] = LOAD(a[i*V]);
	      s[i] = ADD(s[i], v[i]);
	      va   = LOAD(ai[i*V]);
	      v[i] = MUL(s[i], va);
	      v[i] = MUL(v[i], vb);
	      STORE(b[i*V], v[i]);
	    }
          a += ld;
	  b += ld;
	} 

      for (j = 0; j < m - 2 * r - 1; j++)
	{
	  for (i = 0; i<NR; i++)
	    {
	      v[i] = LOAD(a[i*V]);
	      s[i] = ADD(s[i], v[i]);
	      v[i] = LOAD(a_diff[i*V]);
	      s[i] = SUB(s[i], v[i]);
	      va   = LOAD(ai[i*V]);
	      v[i] = MUL(s[i], va);
	      STORE(b[i*V], v[i]);
	    }
          a += ld;
	  b += ld;
	  a_diff += ld;
	};

      for (j = 0; j < r; j++)
	{
	  vb = BROADCAST(b_norm[r-1-j]);
	  for (i = 0; i<NR; i++)
	    {
	      v[i] = LOAD(a_diff[i*V]);
	      s[i] = SUB(s[i], v[i]);
	      va   = LOAD(ai[i*V]);
	      v[i] = MUL(s[i], va);
	      v[i] = MUL(v[i], vb);
	      STORE(b[i*V], v[i]);
	    }
	  b += ld;
	  a_diff += ld;
	  
	}

      a0 += NR*V;
      b0 += NR*V;
      ai += NR*V;
            
    }

  
  rest  = n - (ni<<POW_T);

  if (rest == 0) return;

  ni = rest >> POW_V;  //ni = rest/V


  //based on leading dimension being divisible by V!
  if (rest - (ni << POW_V) > 0)  ni = ni + 1; 
  
  
  a = a0;
  b = b0;
  a_diff = a0;
  
  s[0] = BROADCAST(zero);
  for (i = 1; i<ni; i++)
    {
      s[i] = s[0];
    }

  for (j = 0; j < r; j++)
    {
      for (i = 0; i<ni; i++)
	{
	  v[i] = LOAD(a[i*V]);
	  s[i] = ADD(s[i], v[i]);
	    }
      a += ld;	  
    }
  
  for (j = 0; j < r + 1; j++)
    {
      vb = BROADCAST(b_norm[j]);
      for (i = 0; i<ni; i++)
	{
	  v[i] = LOAD(a[i*V]);
	  s[i] = ADD(s[i], v[i]);
	  va   = LOAD(ai[i*V]);
	  v[i] = MUL(s[i], va);
	  v[i] = MUL(v[i], vb);
	  STORE(b[i*V], v[i]);
	}
      a += ld;
      b += ld;
    } while (j -= 1);
  
  for (j = 0; j < m - 2 * r - 1; j++)
    {
      for (i = 0; i<ni; i++)
	{
	  v[i] = LOAD(a[i*V]);
	  s[i] = ADD(s[i], v[i]);
	  v[i] = LOAD(a_diff[i*V]);
	  s[i] = SUB(s[i], v[i]);
	  va   = LOAD(ai[i*V]);
	  v[i] = MUL(s[i], va);
	  STORE(b[i*V], v[i]);
	}
      a += ld;
      b += ld;
      a_diff += ld;
    }
  
  for (j = 0; j < r; j++)
    {
      vb = BROADCAST(b_norm[r-1-j]);
      for (i = 0; i<ni; i++)
	{
	  v[i] = LOAD(a_diff[i*V]);
	  s[i] = SUB(s[i], v[i]);
	  va   = LOAD(ai[i*V]);
	  v[i] = MUL(s[i], va);
	  v[i] = MUL(v[i], vb);
	  STORE(b[i*V], v[i]);
	}
      b += ld;
      a_diff += ld;
      
    }

  return;
}
