// Additional fixed-control coverage; shares only this check's diagnostic helpers.
#define INTERIOR_NEWTON_NO_MAIN
#include "check.cpp"
#include <Eigen/LU>
#include <nlohmann/json.hpp>
#include <numeric>
using Json=nlohmann::json;
MatrixXd matrix(const Json& j){
 MatrixXd m(j.size(),j.at(0).size());
 for(int i=0;i<m.rows();++i)for(int k=0;k<m.cols();++k)m(i,k)=j.at(i).at(k).get<double>();
 return m;
}
Json read_json(const std::string& path){std::ifstream f(path);require(bool(f),"fixture file");auto j=Json::parse(f,nullptr,false);require(!j.is_discarded(),"fixture JSON");return j;}
struct Design {
 std::string id,family,syntax,source="synthetic";
 spec::BuildOptions options;
 MatrixXd sigma;
 VectorXd mean;
 int factors=0,groups=1;
 bool imbalance=false;
};
std::string measurement(int factors,int per){
 std::string s;for(int j=0;j<factors;++j){s+="f"+std::to_string(j+1)+" =~ ";for(int k=0;k<per;++k){if(k)s+=" + ";s+="x"+std::to_string(j*per+k+1);}s+='\n';}return s;
}
std::vector<Design> designs(){
 std::vector<Design> ds;
 for(int p:{4,8}){
  Design d;d.id="growth"+std::to_string(p);d.family="growth";d.factors=2;
  MatrixXd L(p,2);for(int j=0;j<p;++j){L(j,0)=1;L(j,1)=j;}
  MatrixXd P(2,2);P<<1,.15,.15,.2;
  d.sigma=L*P*L.transpose()+.7*MatrixXd::Identity(p,p);VectorXd means(2);means<<1,.2;d.mean=L*means;
  for(int f=0;f<2;++f){d.syntax+=(f==0?"i =~ ":"s =~ ");for(int j=0;j<p;++j){if(j)d.syntax+=" + ";d.syntax+=std::to_string(f==0?1:j)+"*x"+std::to_string(j+1);}d.syntax+='\n';}
  d.options.meanstructure=true;d.options.int_ov_free=false;d.options.int_lv_free=true;d.options.fixed_x=false;ds.push_back(d);
 }
 {
  Design d;d.id="feedback12";d.family="feedback";d.factors=4;d.syntax=measurement(4,3)+"f3 ~ f4 + f1\nf4 ~ f3 + f2\n";
  MatrixXd L=MatrixXd::Zero(12,4);for(int j=0;j<12;++j)L(j,j/3)=j%3==0?1:.8;
  MatrixXd B=MatrixXd::Zero(4,4);B(2,3)=.2;B(3,2)=.25;B(2,0)=.5;B(3,1)=.4;
  MatrixXd A=(MatrixXd::Identity(4,4)-B).inverse();d.sigma=L*A*A.transpose()*L.transpose()+.7*MatrixXd::Identity(12,12);d.mean=VectorXd::Zero(12);d.options.fixed_x=false;ds.push_back(d);
 }
 for(int mode=0;mode<4;++mode){
  Design d;d.id=std::vector<std::string>{"mg_configural","mg_metric","mg_scalar","mg5_metric_unbalanced"}[mode];d.family="multigroup";d.factors=3;d.groups=mode==3?5:2;d.imbalance=mode==3;
  d.syntax=measurement(3,4);d.options.fixed_x=false;d.options.meanstructure=true;d.options.n_groups=d.groups;
  if(mode>0)d.options.group_equal.push_back(spec::GroupEqual::Loadings);
  if(mode==2)d.options.group_equal.push_back(spec::GroupEqual::Intercepts);
  MatrixXd L=MatrixXd::Zero(12,3);for(int j=0;j<12;++j)L(j,j/4)=j%4==0?1:.8;
  MatrixXd P=MatrixXd::Constant(3,3,.3);P.diagonal().setOnes();d.sigma=L*P*L.transpose()+.7*MatrixXd::Identity(12,12);
  d.mean=VectorXd::LinSpaced(12,.1,1.2);ds.push_back(d);
 }
 return ds;
}
#ifndef INTERIOR_NEWTON_NO_ADVANCED_MAIN
int main(int argc,char** argv){
 require(argc==2,"usage: advanced output.csv");exact_checks();std::ofstream out(argv[1]);require(bool(out),"output");out<<std::setprecision(17);
 out<<"model,family,source,n_base,n_total,group_sizes,rep,units,seed,profile,p,factors,groups,free_parameters,imbalance,rc,evals,interior,status,distance,condition,fmin,fit_ms,hessian_ms,hessian_fd_rel\n";
 auto run=[&](const Design& d,const data::SampleStats& original,int nbase,int replication,unsigned seed){
  auto parsed=parse::Parser::parse(d.syntax);require(bool(parsed),"parse advanced");auto pt=spec::build(*parsed,d.options);require(bool(pt),"build advanced");auto rep=model::build_matrix_rep(*pt);require(bool(rep),"matrix rep advanced");auto ev=model::ModelEvaluator::build(*pt,*rep);require(bool(ev),"evaluator advanced");auto con=estimate::build_eq_constraints(*pt);require(bool(con),"equalities advanced");
  int n=std::accumulate(original.n_obs.begin(),original.n_obs.end(),0);std::string sizes;for(auto ng:original.n_obs){if(!sizes.empty())sizes+=';';sizes+=std::to_string(ng);}
  for(double units:{.1,1.,10.}){
   auto samp=original;for(auto& S:samp.S)S*=units*units;for(auto& m:samp.mean)m*=units;
   auto start=estimate::fabin_start_values(*pt,*rep,samp,{});require(bool(start),"start advanced");auto obj=estimate::ml_objective(*ev,samp);require(bool(obj),"objective advanced");Context ctx{&*obj,&*con};
   for(const auto& pr:validation_profiles){
    VectorXd a=con->contract(*start);nlopt_opt opt=nlopt_create(NLOPT_LD_LBFGS,a.size());require(opt,"nlopt advanced");nlopt_set_min_objective(opt,callback,&ctx);nlopt_set_ftol_rel(opt,pr.ftol);nlopt_set_xtol_rel(opt,pr.xtol);nlopt_set_maxeval(opt,5000);
    double f;auto t=Clock::now();int rc=nlopt_optimize(opt,a.data(),&f);double elapsed=ms(t);int evaluations=nlopt_get_numevals(opt);nlopt_destroy(opt);
    t=Clock::now();auto point=point_at(a,ctx,*pt,*rep,*ev,samp,n);double hms=ms(t),fd=NAN;
    if(units==1&&std::string(pr.name)=="current"&&replication==1){
     VectorXd v=VectorXd::Ones(a.size());v.normalize();VectorXd gp,gm;double h=1e-5;
     obj->f(con->expand(a+h*v),gp);obj->f(con->expand(a-h*v),gm);
     if(point.info.size()&&gp.allFinite()&&gm.allFinite()){
      VectorXd numerical=n*con->reduce_gradient(gp-gm)/(2*h),analytic=point.info*v;
      fd=(numerical-analytic).norm()/(1+analytic.norm());require(fd<1e-5,"advanced information normalization");
     }
    }
    out<<d.id<<','<<d.family<<','<<d.source<<','<<nbase<<','<<n<<','<<sizes<<','<<replication<<','<<units<<','<<seed<<','<<pr.name<<','<<samp.S[0].rows()<<','<<d.factors<<','<<samp.S.size()<<','<<a.size()<<','<<d.imbalance<<','<<rc<<','<<evaluations<<','<<point.interior<<','<<point.audit.status<<','<<point.audit.distance<<','<<point.audit.condition<<','<<point.f<<','<<elapsed<<','<<hms<<','<<fd<<'\n';
   }
  }
  out.flush();
 };
 auto ds=designs();
 for(size_t ci=0;ci<ds.size();++ci){auto& d=ds[ci];
  for(int n:{50,200,1000,10000})for(int r=1;r<=2;++r){
   unsigned seed=986021+ci*100000+13*n+r;std::mt19937 rng(seed);std::normal_distribution<double> normal;data::SampleStats samp;
   for(int g=0;g<d.groups;++g){int p=d.sigma.rows();int ng=d.imbalance?std::max(p+2,n/(g+1)):n;MatrixXd S=d.sigma*(1+.2*g);Eigen::LLT<MatrixXd> chol(S);require(chol.info()==Eigen::Success,"advanced population PD");MatrixXd X(ng,p);for(int i=0;i<ng;++i)for(int j=0;j<p;++j)X(i,j)=normal(rng);X=(X*MatrixXd(chol.matrixL()).transpose()).eval();VectorXd m=X.colwise().mean();MatrixXd centered=X.rowwise()-m.transpose();samp.S.push_back(centered.transpose()*centered/ng);samp.n_obs.push_back(ng);if(d.options.meanstructure)samp.mean.push_back(m+d.mean);}
   run(d,samp,n,r,seed);
  }std::cerr<<"Completed "<<d.id<<'\n';
 }
 for(const std::string id:{"bollen_democracy_sem","hs_3factor_cfa"}){
  std::string path="tests/fixtures/parity/"+id+"/reference.json";auto j=read_json(path);Design d;d.id=id;d.family="published";d.source=path;d.syntax=j.at("model");d.factors=3;d.options.fixed_x=false;d.options.meanstructure=j.at("meanstructure");d.options.auto_cov_y=j.at("auto_cov_y");
  data::SampleStats samp;samp.S.push_back(matrix(j.at("sample_cov").at("values")));samp.n_obs.push_back(j.at("n_obs").get<int>());require(!d.options.meanstructure,"reference mean format");run(d,samp,samp.n_obs[0],1,0);std::cerr<<"Completed "<<id<<'\n';
 }
 const std::string path="tests/fixtures/textbook_corpus/case_exports.json";
 const auto corpus=read_json(path);
 for(const auto& j:corpus.at("cases")){
  Design d;d.id=j.at("case_id");d.family="published_invariance";d.source=path;d.syntax=j.at("model");d.factors=2;d.groups=2;d.options.fixed_x=false;d.options.meanstructure=true;d.options.n_groups=2;
  const auto& data=j.at("data");data::SampleStats samp;
  for(size_t g=0;g<data.at("n_obs").size();++g){samp.n_obs.push_back(data.at("n_obs").at(g).get<int>());samp.S.push_back(matrix(data.at("sample_cov").at(g)));const auto& jm=data.at("sample_mean").at(g);VectorXd m(jm.size());for(int k=0;k<m.size();++k)m(k)=jm.at(k).get<double>();samp.mean.push_back(m);}
  run(d,samp,0,1,0);std::cerr<<"Completed "<<d.id<<'\n';
 }
}

#endif
