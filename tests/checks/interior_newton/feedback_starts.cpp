#define INTERIOR_NEWTON_NO_ADVANCED_MAIN
#include "advanced.cpp"
struct ScaledContext {Context* original; VectorXd scale;};
double scaled_callback(unsigned n,const double* x,double* grad,void* ptr){
 auto& c=*static_cast<ScaledContext*>(ptr);VectorXd z=Eigen::Map<const VectorXd>(x,n);VectorXd a=c.scale.cwiseProduct(z),g;
 double f=c.original->objective->f(c.original->con->expand(a),g);
 if(grad)Eigen::Map<VectorXd>(grad,n)=c.scale.cwiseProduct(c.original->con->reduce_gradient(g));
 return f;
}
int main(int argc,char** argv){
 require(argc==2,"usage: feedback_starts output.csv");exact_checks();auto d=designs()[2];require(d.id=="feedback12","feedback design index");
 auto parsed=parse::Parser::parse(d.syntax);require(bool(parsed),"parse");auto pt=spec::build(*parsed,d.options);require(bool(pt),"build");auto rep=model::build_matrix_rep(*pt);require(bool(rep),"rep");auto ev=model::ModelEvaluator::build(*pt,*rep);require(bool(ev),"ev");auto con=estimate::build_eq_constraints(*pt);require(bool(con),"con");
 require(con->K().isIdentity(),"this scaling experiment assumes no equality reduction");
 std::ofstream out(argv[1]),starts(std::string(argv[1])+".starts.csv");out<<std::setprecision(17);starts<<std::setprecision(17);
 out<<"n,rep,seed,units,arm,start_f,start_gradient_norm,start_psi_min,start_theta_min,start_difference,rc,evals,interior,status,distance,fmin,fit_ms\n";
 starts<<"n,rep,units,parameter,matrix,row,col,native,mapped,normalized_native\n";
 for(int n:{50,200,1000,10000})for(int r=1;r<=2;++r){
  unsigned seed=986021+2*100000+13*n+r;std::mt19937 rng(seed);std::normal_distribution<double> normal;MatrixXd X(n,12);for(int i=0;i<n;++i)for(int j=0;j<12;++j)X(i,j)=normal(rng);
  X=(X*MatrixXd(d.sigma.llt().matrixL()).transpose()).eval();VectorXd mean=X.colwise().mean();MatrixXd centered=X.rowwise()-mean.transpose();data::SampleStats base;base.S={centered.transpose()*centered/n};base.n_obs={n};auto base_start=estimate::fabin_start_values(*pt,*rep,base,{});require(bool(base_start),"base start");
  auto base_obj=estimate::ml_objective(*ev,base);require(bool(base_obj),"base objective");VectorXd base_g;double base_f=base_obj->f(*base_start,base_g);
  for(double units:{.1,1.,10.}){
   auto samp=base;samp.S[0]*=units*units;auto obj=estimate::ml_objective(*ev,samp);require(bool(obj),"objective");Context ctx{&*obj,&*con};auto native=estimate::fabin_start_values(*pt,*rep,samp,{});require(bool(native),"native start");
   VectorXd scale=VectorXd::Ones(base_start->size());
   for(size_t i=0;i<pt->free.size();++i)if(pt->free[i]>0){auto c=rep->cell_for_row[i];if(c.mat==model::MatId::Psi||c.mat==model::MatId::Theta)scale(pt->free[i]-1)=units*units;}
   VectorXd mapped=scale.cwiseProduct(*base_start),g;double mapped_f=obj->f(mapped,g);
   require(std::abs(mapped_f-base_f)<1e-10,"mapped initial objective invariance");
   require((scale.cwiseProduct(g)-base_g).norm()/(1+base_g.norm())<1e-9,"mapped gradient invariance");
   for(size_t i=0;i<pt->free.size();++i)if(pt->free[i]>0){int k=pt->free[i]-1;auto c=rep->cell_for_row[i];starts<<n<<','<<r<<','<<units<<','<<k<<','<<model::to_string(c.mat)<<','<<c.row<<','<<c.col<<','<<(*native)(k)<<','<<mapped(k)<<','<<(*native)(k)/scale(k)<<'\n';}
   for(const std::string arm:{"native","mapped_start","mapped_coordinates"}){
    VectorXd initial=arm=="native"?*native:mapped;VectorXd gi;double fi=obj->f(initial,gi);auto mat=ev->assembled(initial);require(bool(mat),"initial matrices");
    Eigen::SelfAdjointEigenSolver<MatrixXd> psi(mat->blocks[0].Psi),theta(mat->blocks[0].Theta);
    VectorXd a=arm=="mapped_coordinates"?*base_start:initial;ScaledContext scaled{&ctx,scale};
    nlopt_opt opt=nlopt_create(NLOPT_LD_LBFGS,a.size());require(opt,"nlopt");
    if(arm=="mapped_coordinates")nlopt_set_min_objective(opt,scaled_callback,&scaled);else nlopt_set_min_objective(opt,callback,&ctx);
    nlopt_set_ftol_rel(opt,1e-12);nlopt_set_xtol_rel(opt,1e-10);nlopt_set_maxeval(opt,5000);double f;auto t=Clock::now();int rc=nlopt_optimize(opt,a.data(),&f);double elapsed=ms(t);int evals=nlopt_get_numevals(opt);nlopt_destroy(opt);
    if(arm=="mapped_coordinates")a=scale.cwiseProduct(a);
    auto point=point_at(a,ctx,*pt,*rep,*ev,samp,n);
    out<<n<<','<<r<<','<<seed<<','<<units<<','<<arm<<','<<fi<<','<<gi.norm()<<','<<psi.eigenvalues().minCoeff()<<','<<theta.eigenvalues().minCoeff()<<','<<(initial-mapped).cwiseAbs().maxCoeff()<<','<<rc<<','<<evals<<','<<point.interior<<','<<point.audit.status<<','<<point.audit.distance<<','<<point.f<<','<<elapsed<<'\n';
   }
  }
  out.flush();starts.flush();std::cerr<<"Completed n="<<n<<" rep="<<r<<'\n';
 }
}
