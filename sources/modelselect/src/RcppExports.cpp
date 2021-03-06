// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// modelSelection_trans_mat
List modelSelection_trans_mat(arma::mat& mat_pred, arma::vec& obj_pred, arma::mat& Rt, size_t seqLen, size_t K, List& obj_Kmodel, float gamma, arma::vec& v_muLBI, size_t clamp);
RcppExport SEXP _modelselect_modelSelection_trans_mat(SEXP mat_predSEXP, SEXP obj_predSEXP, SEXP RtSEXP, SEXP seqLenSEXP, SEXP KSEXP, SEXP obj_KmodelSEXP, SEXP gammaSEXP, SEXP v_muLBISEXP, SEXP clampSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type mat_pred(mat_predSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type obj_pred(obj_predSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type Rt(RtSEXP);
    Rcpp::traits::input_parameter< size_t >::type seqLen(seqLenSEXP);
    Rcpp::traits::input_parameter< size_t >::type K(KSEXP);
    Rcpp::traits::input_parameter< List& >::type obj_Kmodel(obj_KmodelSEXP);
    Rcpp::traits::input_parameter< float >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type v_muLBI(v_muLBISEXP);
    Rcpp::traits::input_parameter< size_t >::type clamp(clampSEXP);
    rcpp_result_gen = Rcpp::wrap(modelSelection_trans_mat(mat_pred, obj_pred, Rt, seqLen, K, obj_Kmodel, gamma, v_muLBI, clamp));
    return rcpp_result_gen;
END_RCPP
}
// computeR
void computeR(arma::mat& E_obs, arma::mat& Rt, arma::vec& muLBI, size_t clamp);
RcppExport SEXP _modelselect_computeR(SEXP E_obsSEXP, SEXP RtSEXP, SEXP muLBISEXP, SEXP clampSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type E_obs(E_obsSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type Rt(RtSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type muLBI(muLBISEXP);
    Rcpp::traits::input_parameter< size_t >::type clamp(clampSEXP);
    computeR(E_obs, Rt, muLBI, clamp);
    return R_NilValue;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_modelselect_modelSelection_trans_mat", (DL_FUNC) &_modelselect_modelSelection_trans_mat, 9},
    {"_modelselect_computeR", (DL_FUNC) &_modelselect_computeR, 4},
    {NULL, NULL, 0}
};

RcppExport void R_init_modelselect(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
