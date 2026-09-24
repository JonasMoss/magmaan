#define INTERIOR_NEWTON_NO_ADVANCED_MAIN
#include "advanced.cpp"
#include "broad_helpers.hpp"
#include <optional>
int main(int argc,char** argv){
 require(argc==2,"usage: advanced output.csv");exact_checks();std::ofstream checks(std::string(argv[1])+".checks.csv");checks<<"model,n_base,gradient_relative_error\n";std::ofstream out(argv[1]);require(bool(out),"output");out<<std::setprecision(17);
 out<<"model,family,n_base,n_total,rep,seed,units,policy,branch,rc,evals,interior,status,distance,fmin,prep_ms,fit_ms,total_ms,p,groups,free_parameters\n";
 auto run=[&](const Design& d,const data::SampleStats& original,int nbase,int replication,unsigned seed){
  auto begin=Clock::now();auto parsed=parse::Parser::parse(d.syntax);require(bool(parsed),"parse");auto pt=spec::build(*parsed,d.options);require(bool(pt),"build");auto rep=model::build_matrix_rep(*pt);require(bool(rep),"rep");auto ev=model::ModelEvaluator::build(*pt,*rep);require(bool(ev),"ev");auto con=estimate::build_eq_constraints(*pt);require(bool(con),"constraints");auto layout=broad_layout(*pt,*rep,*con);double sharedms=ms(begin);
  int n=std::accumulate(original.n_obs.begin(),original.n_obs.end(),0);
  for(double units:{.1,1.,10.}){auto samp=original;for(auto& S:samp.S)S*=units*units;for(auto& m:samp.mean)m*=units;
   for(const std::string policy:{"baseline","candidate","std_lv"}){
    if(policy=="std_lv"&&!layout.supported){out<<d.id<<','<<d.family<<','<<nbase<<','<<n<<','<<replication<<','<<seed<<','<<units<<','<<policy<<",unsupported,0,0,0,unavailable,nan,nan,0,0,"<<sharedms<<','<<samp.S[0].rows()<<','<<samp.S.size()<<','<<con->n_alpha<<'\n';continue;}
    auto prep=Clock::now();auto start=estimate::fabin_start_values(*pt,*rep,samp,{});require(bool(start),"start");auto obj=estimate::ml_objective(*ev,samp);require(bool(obj),"objective");Context ctx{&*obj,&*con};
    // Build the alternative chart only within the proven transport scope.
    std::optional<spec::LatentStructure> sp;std::optional<model::MatrixRep> sr;std::optional<model::ModelEvaluator> se;std::optional<estimate::EqConstraints> sc;
    if(policy!="baseline"&&layout.supported){auto so=d.options;so.std_lv=true;
     auto built=spec::build(*parsed,so);require(bool(built),"std build");sp.emplace(std::move(*built));
     auto mapped=model::build_matrix_rep(*sp);require(bool(mapped),"std rep");sr.emplace(std::move(*mapped));
     auto evaluator=model::ModelEvaluator::build(*sp,*sr);require(bool(evaluator),"std ev");se.emplace(std::move(*evaluator));
     auto reduced=estimate::build_eq_constraints(*sp);require(bool(reduced),"std con");sc.emplace(std::move(*reduced));
    }
    VectorXd initial=*start;std::string branch="native";
    if(policy!="baseline"&&layout.supported){auto ss=estimate::fabin_start_values(*sp,*sr,samp,{});require(bool(ss),"std start");initial=broad_marker(*ss,*se,*pt,*rep,layout);auto os=estimate::ml_objective(*se,samp);require(bool(os),"std objective");VectorXd g;double f1=obj->f(initial,g),f2=os->f(*ss,g);require(std::abs(f1-f2)<1e-8,"start equivalence");branch="transported";}
    if(policy=="candidate"&&!layout.supported)branch="native_fallback";
    std::optional<optim::ScalarProblem> stdobj;
    if(policy=="std_lv"){auto built=estimate::ml_objective(*se,samp);require(bool(built),"std objective");stdobj.emplace(std::move(*built));}
    VectorXd scale=policy=="candidate"?broad_scale(*pt,*rep,*con,samp):VectorXd::Ones(con->n_alpha);
    VectorXd a=con->contract(initial);Context stdctx;
    if(policy=="std_lv"){require(bool(stdobj),"std fit objective");auto ss=estimate::fabin_start_values(*sp,*sr,samp,{});a=sc->contract(*ss);stdctx={&*stdobj,&*sc};scale=VectorXd::Ones(a.size());}
    BroadMetric metric{policy=="std_lv"?&stdctx:&ctx,scale};a=a.cwiseQuotient(scale);double prepms=ms(prep);
    if(policy=="candidate"&&replication==1&&units==1){VectorXd v=VectorXd::Ones(a.size());v.normalize();VectorXd g(a.size());broad_callback(a.size(),a.data(),g.data(),&metric);double h=1e-6;VectorXd plus=a+h*v,minus=a-h*v;double numerical=(broad_callback(a.size(),plus.data(),nullptr,&metric)-broad_callback(a.size(),minus.data(),nullptr,&metric))/(2*h);double analytical=g.dot(v);double error=std::abs(numerical-analytical)/(1+std::abs(analytical));require(error<1e-5,"scaled reduced gradient");checks<<d.id<<','<<nbase<<','<<error<<'\n';checks.flush();}

    nlopt_opt opt=nlopt_create(NLOPT_LD_LBFGS,a.size());require(opt,"nlopt");nlopt_set_min_objective(opt,broad_callback,&metric);nlopt_set_ftol_rel(opt,1e-12);nlopt_set_xtol_rel(opt,1e-10);nlopt_set_maxeval(opt,5000);double f;auto t=Clock::now();int rc=nlopt_optimize(opt,a.data(),&f);double fitms=ms(t);int evals=nlopt_get_numevals(opt);nlopt_destroy(opt);a=scale.cwiseProduct(a);
    auto auditstart=Clock::now();VectorXd terminal=policy=="std_lv"?con->contract(broad_marker(sc->expand(a),*se,*pt,*rep,layout)):a;
    require((con->A_eq*con->expand(terminal)-con->b_eq).norm()<1e-7,"terminal equalities");auto point=point_at(terminal,ctx,*pt,*rep,*ev,samp,n);double auditms=ms(auditstart);
    if(policy=="std_lv"){VectorXd g;double original=stdobj->f(sc->expand(a),g);require(std::abs(original-point.f)<1e-7,"terminal std objective transport");}
    out<<d.id<<','<<d.family<<','<<nbase<<','<<n<<','<<replication<<','<<seed<<','<<units<<','<<policy<<','<<branch<<','<<rc<<','<<evals<<','<<point.interior<<','<<point.audit.status<<','<<point.audit.distance<<','<<point.f<<','<<prepms<<','<<fitms<<','<<sharedms+prepms+fitms+auditms<<','<<samp.S[0].rows()<<','<<samp.S.size()<<','<<con->n_alpha<<'\n';
   }out.flush();
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
  for(int n:{50,200,1000})for(int r=1;r<=5;++r){
   unsigned seed=9222026+ci*100000+13*n+r;std::mt19937 rng(seed);std::normal_distribution<double> normal;data::SampleStats samp;
   for(int g=0;g<d.groups;++g){int p=d.sigma.rows();int ng=d.imbalance?std::max(p+2,n/(g+1)):n;MatrixXd S=d.sigma*(1+.2*g);Eigen::LLT<MatrixXd> chol(S);require(chol.info()==Eigen::Success,"advanced population PD");MatrixXd X(ng,p);for(int i=0;i<ng;++i)for(int j=0;j<p;++j)X(i,j)=normal(rng);X=(X*MatrixXd(chol.matrixL()).transpose()).eval();VectorXd m=X.colwise().mean();MatrixXd centered=X.rowwise()-m.transpose();samp.S.push_back(centered.transpose()*centered/ng);samp.n_obs.push_back(ng);if(d.options.meanstructure)samp.mean.push_back(m+d.mean);}
   run(d,samp,n,r,seed);
  }std::cerr<<"Completed "<<d.id<<'\n';
 }
 for(const std::string id:{"bollen_democracy_sem","hs_3factor_cfa"}){
  std::string path="cpp/tests/fixtures/parity/"+id+"/reference.json";auto j=read_json(path);Design d;d.id=id;d.family="published";d.source=path;d.syntax=j.at("model");d.factors=3;d.options.fixed_x=false;d.options.meanstructure=j.at("meanstructure");d.options.auto_cov_y=j.at("auto_cov_y");
  data::SampleStats samp;samp.S.push_back(matrix(j.at("sample_cov").at("values")));samp.n_obs.push_back(j.at("n_obs").get<int>());require(!d.options.meanstructure,"reference mean format");run(d,samp,samp.n_obs[0],1,0);std::cerr<<"Completed "<<id<<'\n';
 }
 const std::string path="cpp/tests/fixtures/textbook_corpus/case_exports.json";
 const auto corpus=read_json(path);
 for(const auto& j:corpus.at("cases")){
  Design d;d.id=j.at("case_id");d.family="published_invariance";d.source=path;d.syntax=j.at("model");d.factors=2;d.groups=2;d.options.fixed_x=false;d.options.meanstructure=true;d.options.n_groups=2;
  const auto& data=j.at("data");data::SampleStats samp;
  for(size_t g=0;g<data.at("n_obs").size();++g){samp.n_obs.push_back(data.at("n_obs").at(g).get<int>());samp.S.push_back(matrix(data.at("sample_cov").at(g)));const auto& jm=data.at("sample_mean").at(g);VectorXd m(jm.size());for(int k=0;k<m.size();++k)m(k)=jm.at(k).get<double>();samp.mean.push_back(m);}
  run(d,samp,0,1,0);std::cerr<<"Completed "<<d.id<<'\n';
 }
}

