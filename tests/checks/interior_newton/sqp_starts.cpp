#define INTERIOR_NEWTON_NO_ADVANCED_MAIN
#include "advanced.cpp"
#include "broad_helpers.hpp"
#include <optional>
#include "magmaan/estimate/fit.hpp"
int main(int argc,char** argv){
 require(argc==2||(argc==3&&std::string(argv[2])=="feedback"),"usage sqp output.csv [feedback]");const bool targeted=argc==3;std::ofstream endpoints(std::string(argv[1])+".endpoints.jsonl");exact_checks();std::ofstream out(argv[1]),errors(std::string(argv[1])+".errors.txt");out<<std::setprecision(17);
 out<<"model,n_base,n_total,rep,seed,units,policy,branch,returned,raw_rc,optimizer_status,evals,interior,newton_status,distance,fmin,feasible,covariance_feasible,cone_stationary,cone_residual,nullity,prep_ms,fit_ms,total_ms,p,groups,free_parameters\n";
 auto run=[&](const Design& d,const data::SampleStats& original,int nbase,int replication,unsigned seed){
  if(targeted&&(d.id!="feedback12"||nbase!=200))return;
  auto begin=Clock::now();auto parsed=parse::Parser::parse(d.syntax);require(bool(parsed),"parse");auto pt=spec::build(*parsed,d.options);require(bool(pt),"build");auto rep=model::build_matrix_rep(*pt);require(bool(rep),"rep");auto ev=model::ModelEvaluator::build(*pt,*rep);require(bool(ev),"evaluator");auto con=estimate::build_eq_constraints(*pt);require(bool(con),"constraints");auto layout=broad_layout(*pt,*rep,*con);double sharedms=ms(begin);int n=std::accumulate(original.n_obs.begin(),original.n_obs.end(),0);
  for(double units:{.1,1.,10.}){if(targeted&&units!=10)continue;auto samp=original;for(auto& S:samp.S)S*=units*units;for(auto& m:samp.mean)m*=units;
   for(const std::string policy:{"ordinary_native","ordinary_start","ordinary_scaled","psd_native","psd_start","psd_information"}){
    auto prep=Clock::now();bool psd=policy.starts_with("psd"),native=policy.ends_with("native");auto initial=estimate::fabin_start_values(*pt,*rep,samp,{});require(bool(initial),"start");std::string branch="native";
    if(!native&&layout.supported){auto so=d.options;so.std_lv=true;auto sp=spec::build(*parsed,so);require(bool(sp),"std build");auto sr=model::build_matrix_rep(*sp);require(bool(sr),"std rep");auto se=model::ModelEvaluator::build(*sp,*sr);require(bool(se),"std evaluator");auto ss=estimate::fabin_start_values(*sp,*sr,samp,{});require(bool(ss),"std start");*initial=broad_marker(*ss,*se,*pt,*rep,layout);branch="transported";}else if(!native)branch="native_fallback";
    auto obj=estimate::ml_objective(*ev,samp);require(bool(obj),"objective");Context ctx{&*obj,&*con};VectorXd metric=policy=="ordinary_scaled"?broad_scale(*pt,*rep,*con,samp):VectorXd::Ones(con->n_alpha);double prepms=ms(prep);auto t=Clock::now();VectorXd terminal;int rc=0,status=-1,evals=-1;bool returned=true;
    if(psd){optim::OptimOptions opts;opts.max_iter=5000;opts.nlopt.max_eval=5000;opts.nlopt.ftol_rel=1e-12;opts.nlopt.xtol_rel=1e-10;opts.nlopt.constraint_tol=1e-8;
     estimate::frontier::PsdFitOptions po;po.diagonal_preconditioning=policy=="psd_information";
     auto fitted=estimate::frontier::fit_ml_psd(*pt,*rep,samp,*initial,estimate::Backend::NloptSlsqp,opts,po);
     if(fitted){terminal=con->contract(fitted->theta);evals=fitted->f_evals;status=static_cast<int>(fitted->optimizer_status);}else {returned=false;errors<<d.id<<' '<<nbase<<' '<<units<<' '<<policy<<": "<<fitted.error().detail<<'\n';}
    }else {VectorXd a=con->contract(*initial).cwiseQuotient(metric);BroadMetric context{&ctx,metric};nlopt_opt opt=nlopt_create(NLOPT_LD_SLSQP,a.size());require(opt,"slsqp");nlopt_set_min_objective(opt,broad_callback,&context);nlopt_set_ftol_rel(opt,1e-12);nlopt_set_xtol_rel(opt,1e-10);nlopt_set_maxeval(opt,5000);double f;rc=nlopt_optimize(opt,a.data(),&f);evals=nlopt_get_numevals(opt);nlopt_destroy(opt);terminal=metric.cwiseProduct(a);}
    double fitms=ms(t);
    Json record={{"model",d.id},{"n_base",nbase},{"rep",replication},{"units",units},{"policy",policy},{"returned",returned}};
    record["initial"]=std::vector<double>(initial->data(),initial->data()+initial->size());
    if(returned){VectorXd full=con->expand(terminal);record["theta"]=std::vector<double>(full.data(),full.data()+full.size());}
    endpoints<<record.dump()<<'\n';endpoints.flush();
    Point point;estimate::GeometricStationarityDiagnostics geo;
    if(returned){point=point_at(terminal,ctx,*pt,*rep,*ev,samp,n);VectorXd full=con->expand(terminal),g;double f=obj->f(full,g);if(std::isfinite(f)&&g.allFinite())geo=estimate::audit_geometric_stationarity(full,g,*pt,*ev,*con,estimate::NonlinearEqConstraints{},estimate::Bounds{});}
    out<<d.id<<','<<nbase<<','<<n<<','<<replication<<','<<seed<<','<<units<<','<<policy<<','<<branch<<','<<returned<<','<<rc<<','<<status<<','<<evals<<','<<point.interior<<','<<point.audit.status<<','<<point.audit.distance<<','<<point.f<<','<<geo.feasible<<','<<geo.covariance_feasible<<','<<geo.cone_stationary<<','<<geo.cone_residual_l2<<','<<geo.covariance_nullity<<','<<prepms<<','<<fitms<<','<<sharedms+prepms+ms(t)<<','<<samp.S[0].rows()<<','<<samp.S.size()<<','<<con->n_alpha<<'\n';out.flush();
   }
  }
 };
 auto ds=designs();
 for(int kind=0;kind<8;++kind){Design d;d.id=std::vector<std::string>{"broad_cfa6","broad_cfa24","broad_cfa48","broad_cross12","broad_residual12","broad_weak12","broad_mixed12","broad_equal24"}[kind];d.family="cfa";int p=kind==0?6:(kind==1||kind==7)?24:kind==2?48:12;d.factors=kind==0?1:p/4;d.options.fixed_x=false;d.syntax=measurement(d.factors,p/d.factors);
 MatrixXd L=MatrixXd::Zero(p,d.factors);for(int j=0;j<p;++j)L(j,j/(p/d.factors))=j%(p/d.factors)==0?1:.8;
 MatrixXd P=MatrixXd::Constant(d.factors,d.factors,kind==5?.1:.3);P.diagonal().setConstant(kind==5?.12:1);
 if(kind==3){d.syntax+="f2 =~ x4\nf3 =~ x8\n";L(3,1)=.25;L(7,2)=.25;}
 if(kind==7){d.syntax+="f1 =~ a*x2\nf2 =~ a*x6\n";}
 d.sigma=L*P*L.transpose()+.7*MatrixXd::Identity(p,p);
 if(kind==4){d.syntax+="x1 ~~ x2\nx5 ~~ x6\nx9 ~~ x10\n";for(int j:{0,4,8}){d.sigma(j,j+1)+=.15;d.sigma(j+1,j)+=.15;}}
 if(kind==6){VectorXd s(p);for(int j=0;j<p;++j)s(j)=std::pow(10.,-1.+2.*j/(p-1.));d.sigma=s.asDiagonal()*d.sigma*s.asDiagonal();}
 d.mean=VectorXd::Zero(p);ds.push_back(d);
 }
 for(size_t ci=0;ci<ds.size();++ci){auto& d=ds[ci];
  for(int n:{50,200})for(int r=1;r<=1;++r){
   unsigned seed=9222026+ci*100000+13*n+r;std::mt19937 rng(seed);std::normal_distribution<double> normal;data::SampleStats samp;
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

