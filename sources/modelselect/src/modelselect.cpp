// # utils_ms.cpp
// #
// # Copyright (c) 2018 <pierre.blanchart>
// # Author : Pierre Blanchart
// #
// # This file is part of the "modelselect" package distribution.
// # This program is free software: you can redistribute it and/or modify
// # it under the terms of the GNU General Public License as published by
// # the Free Software Foundation, version 3.
// #
// # This program is distributed in the hope that it will be useful, but
// # WITHOUT ANY WARRANTY; without even the implied warranty of
// # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// # General Public License for more details.
// # You should have received a copy of the GNU General Public License
// # ---------------------------------------------------------------------------


#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
# define ROUND(a) ((int) (((float)(a))+0.5f))
# define MAX_(a, b) (((a) > (b)) ? (a) : (b))



void computeStrategy_trans_mat(int t, size_t seqLen, arma::uvec &path_models, arma::mat &Rt,
                               arma::mat &Qval, arma::umat &Qind, size_t K,
                               size_t nb_models, arma::uvec &index_perm,
                               arma::uvec &index_next, arma::uvec &range_perm, arma::uvec &ind_selected,
                               arma::mat &mat_trans,
                               float gamma) {

  size_t nb_perm = Qval.n_rows;
  size_t nb_next = ((int) ((((float) index_next.n_elem) + 0.5f) / ((float) nb_perm)));

  arma::vec Rs(nb_models);
  arma::vec Qs(nb_perm);
  arma::uvec index_min_s(nb_perm);

  Rs = Rt.col(seqLen-1);
  Qval.col(seqLen-1) = Rs.elem(ind_selected);

  for (int s=seqLen-2; s>=t; s--) {
    // compute possible transitions
    Qs = Qval.col(s+1);
    arma::mat temp_s(gamma*Qs.elem(index_next));
    temp_s.reshape(nb_perm, nb_next);
    temp_s %= mat_trans;
    Rs = Rt.col(s);
    temp_s.each_col() += Rs.elem(ind_selected);

    // backward update
    index_min_s = arma::index_max(temp_s, 1)*nb_perm + range_perm;
    Qind.col(s) = index_next(index_min_s);
    Qval.col(s) = temp_s(index_min_s);
  }

  // compute optimal trajectory from instant t
  arma::uvec path_ind(seqLen);
  path_ind(t) = arma::index_max(Qval.col(t));
  for (size_t s=t+1; s < seqLen; s++) path_ind(s) = Qind(path_ind(s-1), s-1);
  path_models(arma::span(t, seqLen-1)) = index_perm( (K-1)*nb_perm + path_ind(arma::span(t, seqLen-1)) );

}



void updateRt_trans_mat(size_t t, size_t seqLen, arma::rowvec &Rwd_t, arma::uvec &path_models,
                        size_t K, size_t nb_models, size_t nb_perm, arma::uvec &index_perm,
                        arma::uvec &index_next, arma::uvec &range_perm,
                        float gamma, arma::uvec &ind_selected, arma::mat &mat_trans,
                        arma::mat &Rt, arma::mat &Qval, arma::umat &Qind,
                        arma::vec &muLBI, size_t clamp) {

  size_t range_t = std::min(t+clamp, seqLen-1);
  arma::mat update_factor = muLBI(arma::span(0, range_t-t))*Rwd_t;
  Rt(arma::span::all, arma::span(t, range_t)) %= (1 + update_factor.t());

  if (t < seqLen-1) {
    computeStrategy_trans_mat(t, seqLen, path_models, Rt, Qval, Qind, K, nb_models, index_perm, index_next,
                              range_perm, ind_selected, mat_trans, gamma);
  }

}



//' @export
// [[Rcpp::export]]
List modelSelection_trans_mat(arma::mat &mat_pred, arma::vec &obj_pred, arma::mat &Rt, size_t seqLen,
                              size_t K, List &obj_Kmodel, float gamma,
                              arma::vec &v_muLBI, size_t clamp) {

  // printf("Entering model selection ...\n");
  size_t len_obj = obj_pred.n_elem; // length of observations == number of successive strategy readjustments to perform
  size_t nb_models = mat_pred.n_cols;
  arma::mat E_obs = abs(mat_pred(arma::span(0, len_obj-1), arma::span::all) - arma::repmat(obj_pred, 1, nb_models));

  arma::uvec index_perm = as<arma::uvec>(obj_Kmodel["perm"]) - 1;
  arma::uvec index_next = as<arma::uvec>(obj_Kmodel["ind.next"]) - 1;
  arma::uvec ind_selected = as<arma::uvec>(obj_Kmodel["ind.selected"]) - 1;
  arma::mat mat_trans = as<arma::mat>(obj_Kmodel["mat.trans"]);

  arma::rowvec E_t, Rwd_t; // (1 x nb_models)

  // allocate structures
  size_t nb_perm = ROUND(((float) index_perm.n_elem) / ((float) K));
  arma::uvec range_perm = arma::conv_to<arma::uvec>::from( arma::regspace(0, 1, nb_perm-1) );
  arma::mat Qval(nb_perm, seqLen);
  arma::umat Qind(nb_perm, seqLen);

  arma::uvec path_models_t(seqLen);
  arma::mat mat_pred_select(seqLen, len_obj, arma::fill::zeros);
  arma::mat path_models(seqLen, len_obj, arma::fill::zeros);

  arma::uvec range_seqLen = arma::conv_to<arma::uvec>::from( arma::regspace(0, 1, seqLen-1) );

  // performs "len_obj" successive strategy readjustments
  float E_t_max;
  for (size_t t=0; t < len_obj; t++) {

    // printf("Readjusting strategy at instant t=%d ...\n", (int)t);
    E_t = E_obs.row(t);
    E_t_max = E_t.max();
    Rwd_t = E_t_max-E_t;

    updateRt_trans_mat(t, seqLen, Rwd_t, path_models_t, K,
                       nb_models, nb_perm, index_perm, index_next,
                       range_perm, gamma, ind_selected, mat_trans,
                       Rt, Qval, Qind, v_muLBI, clamp);

    mat_pred_select(arma::span(t, seqLen-1), t) = mat_pred( path_models_t(arma::span(t, seqLen-1))*seqLen +
      range_seqLen(arma::span(t, seqLen-1))
    );

    for (size_t u=t; u < seqLen; u++) path_models(u, t) = path_models_t(u)+1; // R indexation

  }

  return List::create(_["mat_pred"]=mat_pred_select, _["mat_index"]=path_models);

}




//' @export
// [[Rcpp::export]]
void computeR(arma::mat &E_obs, arma::mat &Rt, arma::vec &muLBI, size_t clamp) {

  size_t seqLen = E_obs.n_rows;

  arma::rowvec E_t, Rwd_t; // (1 x nb_models)
  float E_t_max;
  for (size_t t=0; t < seqLen; t++) {

    // printf("Readjusting strategy at instant t=%d ...\n", (int)t);
    E_t = E_obs.row(t);
    E_t_max = E_t.max();
    Rwd_t = E_t_max-E_t;

    size_t range_t = std::min(t+clamp, seqLen-1);
    arma::mat update_factor = muLBI(arma::span(0, range_t-t))*Rwd_t;
    Rt(arma::span::all, arma::span(t, range_t)) %= (1 + update_factor.t());

  }

}


