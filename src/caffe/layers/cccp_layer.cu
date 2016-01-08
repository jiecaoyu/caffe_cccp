// Copyright 2014 Min Lin

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CCCPPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      //vector<Blob<Dtype>*>* top) {
     const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const int weight_offset = NUM_OUTPUT_*CHANNEL_/GROUP_;
  const int bottom_group_offset = REST_*CHANNEL_/GROUP_;
  const int top_group_offset = REST_*NUM_OUTPUT_;

  for (int n = 0; n < NUM_; ++n) {
    for (int g = 0; g < GROUP_; ++g) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, NUM_OUTPUT_, REST_, CHANNEL_/GROUP_, (Dtype)1.,
      weight + g*weight_offset, 
      bottom_data + bottom[0]->offset(n)+g*bottom_group_offset, 
      (Dtype)0., 
      top_data + top[0]->offset(n)+g*top_group_offset);
    }
    if (biasterm_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, GROUP_*NUM_OUTPUT_, REST_, 1, (Dtype)1.,
      this->blobs_[1]->gpu_data(), 
      reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()), (Dtype)1.,
      top_data + top[0]->offset(n));
    }
  }
}

template <typename Dtype>
void CCCPPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      //const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* bias_diff = NULL;
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  const int weight_offset = NUM_OUTPUT_*CHANNEL_/GROUP_;
  const int bottom_group_offset = REST_*CHANNEL_/GROUP_;
  const int top_group_offset = REST_*NUM_OUTPUT_;

  // Gradient w.r.t. bias
  if (biasterm_) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    CUDA_CHECK(cudaMemset(bias_diff, 0, sizeof(Dtype) * this->blobs_[1]->count()));
    for (int n = 0; n < NUM_; ++n) {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, NUM_OUTPUT_*GROUP_, REST_, (Dtype)1., 
	top_diff + top[0]->offset(n),
        reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
        (Dtype)1.,
        bias_diff);
    }
  }
  
  CUDA_CHECK(cudaMemset(weight_diff, 0, sizeof(Dtype) * this->blobs_[0]->count()));
  for (int n = 0; n < NUM_; ++n) {
    // The gradient will be accumulated
    for (int g = 0; g < GROUP_; ++g) {
      // Gradient with respect to weight
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, NUM_OUTPUT_, CHANNEL_/GROUP_, REST_, (Dtype)1.,
        top_diff + top[0]->offset(n) + g*top_group_offset, 
        bottom_data + bottom[0]->offset(n) + g*bottom_group_offset, 
        (Dtype)1.,
        weight_diff + g*weight_offset);
      // Gradient w.r.t. bottom data if necessary
      if (propagate_down[0]) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, CHANNEL_/GROUP_, REST_, NUM_OUTPUT_, (Dtype)1.,
            weight + g*weight_offset,
            top_diff + top[0]->offset(n) + g*top_group_offset, 
            (Dtype)0.,
            bottom_diff + bottom[0]->offset(n) + g*bottom_group_offset);
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CCCPPoolingLayer);

}
