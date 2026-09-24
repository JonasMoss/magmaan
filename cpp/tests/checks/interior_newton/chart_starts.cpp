#define INTERIOR_NEWTON_NO_ADVANCED_MAIN
#include "advanced.cpp"
struct MetricContext {Context* ctx;VectorXd scale;};
double metric_callback(unsigned n,const double* x,double* grad,void* ptr){
 auto& c=*static_cast<MetricContext*>(ptr);VectorXd a=c.scale.cwiseProduct(Eigen::Map<const VectorXd>(x,n)),g;
 double f=c.ctx->objective->f(c.ctx->con->expand(a),g);
 if(grad)Eigen::Map<VectorXd>(grad,n)=c.scale.cwiseProduct(c.ctx->con->reduce_gradient(g));return f;
}
VectorXd transport(const VectorXd& x,model::ModelEvaluator& from,const spec::LatentStructure& target,const model::MatrixRep& rep,bool to_std,int per){
 auto m=from.assembled(x);require(bool(m),"transport assembly");auto b=m->blocks[0];int k=b.Psi.rows();VectorXd h(k);
 for(int j=0;j<k;++j){
  if(to_std){require(b.Psi(j,j)>0,"transport positive disturbance");h(j)=std::sqrt(b.Psi(j,j));}
  else {require(std::abs(b.Lambda(j*per,j))>1e-12,"transport nonzero marker");h(j)=1/b.Lambda(j*per,j);}
 }
 b.Lambda=b.Lambda*h.asDiagonal();b.Beta=h.cwiseInverse().asDiagonal()*b.Beta*h.asDiagonal();b.Psi=h.cwiseInverse().asDiagonal()*b.Psi*h.cwiseInverse().asDiagonal();
 VectorXd out=VectorXd::Zero(target.n_free());
 for(size_t i=0;i<target.free.size();++i){auto c=rep.cell_for_row[i];if(!c.used)continue;double value=0;
  switch(c.mat){case model::MatId::Lambda:value=b.Lambda(c.row,c.col);break;case model::MatId::Beta:value=b.Beta(c.row,c.col);break;case model::MatId::Psi:value=b.Psi(c.row,c.col);break;case model::MatId::Theta:value=b.Theta(c.row,c.col);break;default:require(false,"covariance-only transport");}
  if(target.free[i]>0)out(target.free[i]-1)=value;else require(std::abs(value-target.fixed_value[i])<1e-9,"transport fixed constraint");
 }return out;
}
int main(int argc,char** argv){
 require(argc==2,"usage chart_starts output.csv");exact_checks();std::ofstream out(argv[1]);out<<std::setprecision(17);
 out<<"model,p,factors,n,rep,seed,units,start_origin,chart,metric,start_f,start_invariance_error,rc,evals,interior,status,distance,chart_distance,fmin,fit_ms\n";
 std::vector<Design> ds;
 for(int index=0;index<4;++index){Design d;int p=index==0?6:index==1?24:12;d.id=std::vector<std::string>{"cfa6","cfa24","weakmarker12","mixed12"}[index];d.factors=index==0?1:3;d.syntax=measurement(d.factors,p/d.factors);d.options.fixed_x=false;
  MatrixXd L=MatrixXd::Zero(p,d.factors);for(int j=0;j<p;++j)L(j,j/(p/d.factors))=j%(p/d.factors)==0?(index==2?.3:1):.8;
  MatrixXd P=MatrixXd::Constant(d.factors,d.factors,.3);P.diagonal().setOnes();d.sigma=L*P*L.transpose()+.7*MatrixXd::Identity(p,p);
  if(index==3){VectorXd scale(p);for(int j=0;j<p;++j)scale(j)=std::pow(10.,-1.+2.*j/(p-1.));d.sigma=scale.asDiagonal()*d.sigma*scale.asDiagonal();}ds.push_back(d);
 }ds.push_back(designs()[2]);
 for(size_t ci=0;ci<ds.size();++ci){auto& d=ds[ci];int p=d.sigma.rows(),per=p/d.factors;
  auto parsed=parse::Parser::parse(d.syntax);require(bool(parsed),"parse");auto opts=d.options;auto pm=spec::build(*parsed,opts);opts.std_lv=true;auto ps=spec::build(*parsed,opts);require(bool(pm)&&bool(ps),"build charts");auto rm=model::build_matrix_rep(*pm),rs=model::build_matrix_rep(*ps);require(bool(rm)&&bool(rs),"reps");auto em=model::ModelEvaluator::build(*pm,*rm),es=model::ModelEvaluator::build(*ps,*rs);require(bool(em)&&bool(es),"evaluators");auto cm=estimate::build_eq_constraints(*pm),cs=estimate::build_eq_constraints(*ps);require(bool(cm)&&bool(cs)&&cm->K().isIdentity()&&cs->K().isIdentity(),"identity reduction");
  for(int n:{50,200,1000})for(int r=1;r<=2;++r){unsigned seed=220926+100000*ci+13*n+r;std::mt19937 rng(seed);std::normal_distribution<double> normal;MatrixXd X(n,p);for(int i=0;i<n;++i)for(int j=0;j<p;++j)X(i,j)=normal(rng);X=(X*MatrixXd(d.sigma.llt().matrixL()).transpose()).eval();VectorXd mean=X.colwise().mean();MatrixXd centered=X.rowwise()-mean.transpose();MatrixXd S=centered.transpose()*centered/n;
   for(double units:{.1,1.,10.}){data::SampleStats samp;samp.S={units*units*S};samp.n_obs={n};auto om=estimate::ml_objective(*em,samp),os=estimate::ml_objective(*es,samp);require(bool(om)&&bool(os),"objectives");Context ctxm{&*om,&*cm},ctxs{&*os,&*cs};auto sm=estimate::fabin_start_values(*pm,*rm,samp,{}),ss=estimate::fabin_start_values(*ps,*rs,samp,{});require(bool(sm)&&bool(ss),"native starts");
    VectorXd metric=VectorXd::Ones(pm->n_free()),sd=samp.S[0].diagonal().array().sqrt();
    for(size_t i=0;i<pm->free.size();++i)if(pm->free[i]>0){auto c=rm->cell_for_row[i];double scale=1;switch(c.mat){case model::MatId::Lambda:scale=sd(c.row)/sd(c.col*per);break;case model::MatId::Theta:scale=sd(c.row)*sd(c.col);break;case model::MatId::Psi:scale=sd(c.row*per)*sd(c.col*per);break;case model::MatId::Beta:scale=sd(c.row*per)/sd(c.col*per);break;default:require(false,"metric supported cell");}metric(pm->free[i]-1)=scale;}
    for(int origin=0;origin<2;++origin){VectorXd marker=origin==0?*sm:transport(*ss,*es,*pm,*rm,false,per);VectorXd standard=origin==1?*ss:transport(*sm,*em,*ps,*rs,true,per);VectorXd gm,gs;double fm=om->f(marker,gm),fs=os->f(standard,gs);double err=std::abs(fm-fs);require(err<1e-9,"same starting objective");
     auto back=transport(standard,*es,*pm,*rm,false,per);require((back-marker).norm()/(1+marker.norm())<1e-10,"chart roundtrip");
     for(int arm=0;arm<3;++arm){bool st=arm==1;auto& ctx=st?ctxs:ctxm;auto& pt=st?*ps:*pm;auto& rep=st?*rs:*rm;auto& ev=st?*es:*em;VectorXd scale=arm==2?metric:VectorXd::Ones(pt.n_free());VectorXd a=(st?standard:marker).cwiseQuotient(scale);MetricContext mc{&ctx,scale};nlopt_opt opt=nlopt_create(NLOPT_LD_LBFGS,a.size());require(opt,"nlopt");nlopt_set_min_objective(opt,metric_callback,&mc);nlopt_set_ftol_rel(opt,1e-12);nlopt_set_xtol_rel(opt,1e-10);nlopt_set_maxeval(opt,5000);double f;auto t=Clock::now();int rc=nlopt_optimize(opt,a.data(),&f);double elapsed=ms(t);int evals=nlopt_get_numevals(opt);nlopt_destroy(opt);a=scale.cwiseProduct(a);auto chart=point_at(a,ctx,pt,rep,ev,samp,n);VectorXd common=st?transport(a,*es,*pm,*rm,false,per):a;auto audit=point_at(common,ctxm,*pm,*rm,*em,samp,n);require(std::abs(chart.f-audit.f)<1e-8,"terminal objective transport");
      out<<d.id<<','<<p<<','<<d.factors<<','<<n<<','<<r<<','<<seed<<','<<units<<','<<(origin==0?"marker_native":"std_native")<<','<<(st?"std_lv":"marker")<<','<<(arm==2?"data_scaled":"identity")<<','<<fm<<','<<err<<','<<rc<<','<<evals<<','<<audit.interior<<','<<audit.audit.status<<','<<audit.audit.distance<<','<<chart.audit.distance<<','<<audit.f<<','<<elapsed<<'\n';
     }
    }
   }out.flush();
  }std::cerr<<"Completed "<<d.id<<'\n';
 }
}
