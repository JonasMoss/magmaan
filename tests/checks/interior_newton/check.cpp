// Advisory control study: direct NLopt L-BFGS, canonical SEM objective/information.
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <nlopt.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/nl_constraints.hpp"
#include "magmaan/estimate/diagnostics.hpp"
#include "magmaan/inference/inference.hpp"
using namespace magmaan;
using Eigen::MatrixXd; using Eigen::VectorXd;
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
struct Newton {
  std::string status="unavailable";
  double distance=NAN, edmtotal=NAN, relative_solve=NAN, condition=NAN, step_max=NAN;
};
Newton assess(const VectorXd& G,const MatrixXd& I) {
 Newton a;
 if(!G.allFinite()||!I.allFinite()||I.rows()!=G.size()||I.cols()!=G.size())return a;
 if((I.diagonal().array()<=0).any()){a.status="nonpositive_curvature";return a;}
 VectorXd scale=I.diagonal().array().sqrt().inverse();
 MatrixXd C=scale.asDiagonal()*I*scale.asDiagonal(); C=(.5*(C+C.transpose())).eval();
 Eigen::SelfAdjointEigenSolver<MatrixXd> eig(C);
 if(eig.info()!=Eigen::Success||eig.eigenvalues().minCoeff()<=0){a.status="nonpositive_curvature";return a;}
 a.condition=eig.eigenvalues().maxCoeff()/eig.eigenvalues().minCoeff();
 // Numerical safeguard only; not an identification test or tuned scientific cutoff.
 if(a.condition>1e12){a.status="ill_conditioned";return a;}
 Eigen::LLT<MatrixXd> chol(C);if(chol.info()!=Eigen::Success)return a;
 VectorXd b=scale.asDiagonal()*G,y=chol.solve(b),delta=-(scale.asDiagonal()*y);
 a.relative_solve=(C*y-b).norm()/(C.norm()*y.norm()+b.norm()+1e-300);
 if(a.relative_solve>1e-10){a.status="solve_unreliable";return a;}
 double d2=b.dot(y);if(d2<0)return a;
 a.distance=std::sqrt(d2);a.edmtotal=.5*d2;a.step_max=delta.cwiseAbs().maxCoeff();a.status="available";return a;
}
void require(bool b,const char* s){if(!b){std::cerr<<s<<'\n';std::exit(2);}}
void exact_checks(){
 MatrixXd H(2,2);H<<4,1,1,2;VectorXd e(2);e<<.1,-.2;VectorXd g=H*e;
 auto a=assess(g,H);require(a.status=="available"&&std::abs(a.distance*a.distance-e.dot(H*e))<1e-13,"quadratic");
 MatrixXd T(2,2);T<<.01,2,0,100;auto b=assess(T.transpose()*g,T.transpose()*H*T);
 require(std::abs(a.distance-b.distance)<1e-12,"affine invariance");
 auto c=assess(100*g,100*H);require(std::abs(c.distance-10*a.distance)<1e-12,"total likelihood scaling");
 H(1,1)=-1;require(assess(g,H).status=="nonpositive_curvature","indefinite curvature");
 H<<1,1,1,1;require(assess(g,H).status!="available","singular curvature");
}
struct Context {const optim::ScalarProblem* objective;const estimate::EqConstraints* con;};
double callback(unsigned n,const double* x,double* grad,void* ptr){
 auto& c=*static_cast<Context*>(ptr);VectorXd a=Eigen::Map<const VectorXd>(x,n),g;
 double f=c.objective->f(c.con->expand(a),g);
 if(grad)Eigen::Map<VectorXd>(grad,n)=c.con->reduce_gradient(g);
 return f;
}
struct Profile{const char* name;double ftol,xtol,tolg;unsigned memory;};
const std::vector<Profile> profiles={
 {"current",1e-10,1e-7,0,0},{"f12",1e-12,1e-7,0,0},{"f14",1e-14,1e-7,0,0},
 {"x10",1e-10,1e-10,0,0},{"f12_x10",1e-12,1e-10,0,0},
 {"grad10",0,0,1e-10,0},{"grad12",0,0,1e-12,0},
 {"memory10",1e-12,1e-10,0,10},{"memory20",1e-12,1e-10,0,20}};
struct Case{const char* name;int p,factors;bool weak,equal,means,structural,misspecified;};
const std::vector<Case> cases={
 {"cfa4",4,1,false,false,false,false,false},
 {"cfa12",12,1,false,false,false,false,false},
 {"cfa8",8,2,false,false,false,false,false},
 {"weak8",8,2,true,false,false,false,false},
 {"equal8_means",8,2,false,true,true,false,false},
 {"structural8",8,2,false,false,true,true,false},
 {"misspecified4",4,1,false,false,false,false,true}};
int main(int argc,char** argv){
 require(argc==2,"usage: check output.csv");exact_checks();std::ofstream out(argv[1]);require(bool(out),"output");
 out<<std::setprecision(17)<<"model,n,rep,units,seed,profile,ftol,xtol,tolg,memory,rc,evals,fmin,old_residual,interior,status,distance,edm_total,condition,solve_error,step_raw_max,fit_ms,hessian_audit_ms,hessian_fd_rel\n";
 for(size_t ci=0;ci<cases.size();++ci){const auto& c=cases[ci];
  std::string syntax;int per=c.p/c.factors;
  for(int j=0;j<c.factors;++j){syntax+="f"+std::to_string(j+1)+" =~ ";for(int k=0;k<per;++k){if(k)syntax+=" + ";if(c.equal&&k)syntax+="a"+std::to_string(k)+"*";syntax+="x"+std::to_string(j*per+k+1);}syntax+='\n';}
  if(c.structural)syntax+="f2 ~ f1\n";
  auto parsed=parse::Parser::parse(syntax);require(bool(parsed),"parse");spec::BuildOptions bo;bo.meanstructure=c.means;bo.fixed_x=false;
  auto pt=spec::build(*parsed,bo);require(bool(pt),"build");auto rep=model::build_matrix_rep(*pt);require(bool(rep),"rep");
  auto ev=model::ModelEvaluator::build(*pt,*rep);require(bool(ev),"evaluator");auto con=estimate::build_eq_constraints(*pt);require(bool(con),"constraints");
  MatrixXd L=MatrixXd::Zero(c.p,c.factors);for(int j=0;j<c.factors;++j)for(int k=0;k<per;++k)L(j*per+k,j)=k==0?1:(.85-.025*k);
  MatrixXd P=MatrixXd::Identity(c.factors,c.factors)*(c.weak?.12:1);
  if(c.factors==2)P(0,1)=P(1,0)=c.weak?.10:.4;
  if(c.structural){P(0,1)=P(1,0)=.3;P(1,1)=1.09;}
  MatrixXd Sigma=L*P*L.transpose()+MatrixXd::Identity(c.p,c.p)*.7;
  if(c.misspecified)Sigma(0,1)=Sigma(1,0)=Sigma(0,1)+.15;
  MatrixXd root=Sigma.llt().matrixL();
  for(int n:{100,1000,100000})for(int r=1;r<=3;++r){
   unsigned seed=260922+100000*ci+13*n+r;std::mt19937 rng(seed);std::normal_distribution<double> norm;
   MatrixXd X(n,c.p);for(int i=0;i<n;++i)for(int j=0;j<c.p;++j)X(i,j)=norm(rng);
   X=(X*root.transpose()).eval();VectorXd mean=X.colwise().mean();
   MatrixXd centered=X.rowwise()-mean.transpose();MatrixXd S=centered.transpose()*centered/n;
   if(c.means)for(int j=0;j<c.p;++j)mean(j)+=.2*(j+1);
   for(double units:{.1,1.,10.}){
    data::SampleStats samp;samp.S={units*units*S};samp.n_obs={n};if(c.means)samp.mean={units*mean};
    auto start=estimate::fabin_start_values(*pt,*rep,samp,{});require(bool(start),"start");
    auto obj=estimate::ml_objective(*ev,samp);require(bool(obj),"objective");Context ctx{&*obj,&*con};
    for(const auto& pr:profiles){VectorXd a=con->contract(*start);nlopt_opt opt=nlopt_create(NLOPT_LD_LBFGS,a.size());require(opt,"nlopt");
     nlopt_set_min_objective(opt,callback,&ctx);nlopt_set_ftol_rel(opt,pr.ftol);nlopt_set_xtol_rel(opt,pr.xtol);nlopt_set_maxeval(opt,5000);
     if(pr.tolg>0)nlopt_set_param(opt,"tolg",pr.tolg);if(pr.memory)nlopt_set_vector_storage(opt,pr.memory);
     double f;auto t=Clock::now();int rc=nlopt_optimize(opt,a.data(),&f);double fitms=ms(t);int evals=nlopt_get_numevals(opt);nlopt_destroy(opt);
     estimate::Estimates est;est.theta=con->expand(a);VectorXd g;f=obj->f(est.theta,g);est.fmin=f;
     double old=NAN;bool interior=false;Newton audit;double hms=NAN,fdrel=NAN;
     if(std::isfinite(f)&&g.allFinite()){
      auto geo=estimate::audit_geometric_stationarity(est.theta,g,*pt,*ev,*con,estimate::NonlinearEqConstraints{},estimate::Bounds{});old=geo.ambient_residual_l2;
      auto mat=ev->assembled(est.theta);interior=bool(mat);
      if(mat)for(auto& b:mat->blocks)for(const MatrixXd* M:{&b.Psi,&b.Theta}){
       Eigen::SelfAdjointEigenSolver<MatrixXd> es(*M);if(es.info()!=Eigen::Success||es.eigenvalues().minCoeff()<=1e-8*es.eigenvalues().cwiseAbs().maxCoeff())interior=false;
      }
      t=Clock::now();auto info=inference::information_observed_analytic(*pt,*rep,samp,est);
      if(info){MatrixXd I=con->K().transpose()*(*info)*con->K();VectorXd G=n*con->reduce_gradient(g);audit=assess(G,I);}
      hms=ms(t);
      if(info&&pr.name==std::string("current")&&n==1000&&r==1&&units==1){
       // Independent directional finite difference of the canonical gradient.
       VectorXd direction=VectorXd::Ones(a.size());direction.normalize();double h=1e-5;VectorXd gp,gm;
       obj->f(con->expand(a+h*direction),gp);obj->f(con->expand(a-h*direction),gm);
       VectorXd fd=n*con->reduce_gradient(gp-gm)/(2*h),analytic=con->K().transpose()*(*info)*con->K()*direction;
       fdrel=(fd-analytic).norm()/(1+analytic.norm());require(fdrel<1e-5,"analytic Hessian normalization/direction");
      }
     }
     out<<c.name<<','<<n<<','<<r<<','<<units<<','<<seed<<','<<pr.name<<','<<pr.ftol<<','<<pr.xtol<<','<<pr.tolg<<','<<pr.memory<<','<<rc<<','<<evals<<','<<f<<','<<old<<','<<interior<<','<<audit.status<<','<<audit.distance<<','<<audit.edmtotal<<','<<audit.condition<<','<<audit.relative_solve<<','<<audit.step_max<<','<<fitms<<','<<hms<<','<<fdrel<<'\n';
    }
   }
  }
  std::cerr<<"Completed "<<c.name<<'\n';out.flush();
 }
}
