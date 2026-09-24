// Advisory policy frozen before the broad validation run.
struct BroadMetric {Context* ctx;VectorXd scale;};
double broad_callback(unsigned n,const double* x,double* gradient,void* ptr){
 auto& c=*static_cast<BroadMetric*>(ptr);VectorXd a=c.scale.cwiseProduct(Eigen::Map<const VectorXd>(x,n)),g;
 double f=c.ctx->objective->f(c.ctx->con->expand(a),g);
 if(gradient)Eigen::Map<VectorXd>(gradient,n)=c.scale.cwiseProduct(c.ctx->con->reduce_gradient(g));return f;
}
struct Layout {bool supported=true;std::vector<std::vector<int>> markers;};
Layout broad_layout(const spec::LatentStructure& pt,const model::MatrixRep& rep,const estimate::EqConstraints& con){
 Layout l;l.supported=!con.active();
 for(auto d:rep.dims)l.markers.push_back(std::vector<int>(d.n_latent,-1));
 for(size_t i=0;i<pt.free.size();++i){auto c=rep.cell_for_row[i];if(!c.used||pt.free[i]>0)continue;double v=pt.fixed_value[i];
  if(c.mat==model::MatId::Lambda&&v!=0){if(v!=1||l.markers[c.block][c.col]!=-1)l.supported=false;l.markers[c.block][c.col]=c.row;}
  else if(v!=0)l.supported=false;
 }
 for(auto& b:l.markers)for(int row:b)if(row<0)l.supported=false;
 return l;
}
VectorXd broad_marker(const VectorXd& x,model::ModelEvaluator& standard,const spec::LatentStructure& target,const model::MatrixRep& rep,const Layout& layout){
 auto m=standard.assembled(x);require(bool(m),"std assembly");VectorXd out=VectorXd::Zero(target.n_free());
 for(size_t b=0;b<m->blocks.size();++b){auto& v=m->blocks[b];VectorXd h(v.Psi.rows());for(int j=0;j<h.size();++j){double marker=v.Lambda(layout.markers[b][j],j);require(std::abs(marker)>1e-12,"nonzero chart marker");h(j)=1/marker;}
  v.Lambda=v.Lambda*h.asDiagonal();v.Beta=h.cwiseInverse().asDiagonal()*v.Beta*h.asDiagonal();v.Psi=h.cwiseInverse().asDiagonal()*v.Psi*h.cwiseInverse().asDiagonal();if(v.Alpha.size())v.Alpha=h.cwiseInverse().asDiagonal()*v.Alpha;
 }
 for(size_t i=0;i<target.free.size();++i){auto c=rep.cell_for_row[i];if(!c.used)continue;auto& b=m->blocks[c.block];double v=0;switch(c.mat){case model::MatId::Lambda:v=b.Lambda(c.row,c.col);break;case model::MatId::Beta:v=b.Beta(c.row,c.col);break;case model::MatId::Psi:v=b.Psi(c.row,c.col);break;case model::MatId::Theta:v=b.Theta(c.row,c.col);break;case model::MatId::Nu:v=b.Nu(c.row);break;case model::MatId::Alpha:v=b.Alpha(c.row);break;}
  if(target.free[i]>0)out(target.free[i]-1)=v;else require(std::abs(v-target.fixed_value[i])<1e-8,"fixed value preserved");
 }return out;
}
VectorXd broad_scale(const spec::LatentStructure& pt,const model::MatrixRep& rep,const estimate::EqConstraints& con,const data::SampleStats& samp){
 std::vector<VectorXd> obs,latent;
 for(size_t b=0;b<samp.S.size();++b){obs.push_back(samp.S[b].diagonal().array().sqrt());latent.push_back(VectorXd::Constant(rep.dims[b].n_latent,obs.back().mean()));}
 // A fixed loading supplies a natural latent unit; use the first nonzero one.
 std::vector<std::vector<bool>> assigned;for(auto v:latent)assigned.emplace_back(v.size(),false);
 for(size_t i=0;i<pt.free.size();++i){auto c=rep.cell_for_row[i];if(c.used&&c.mat==model::MatId::Lambda&&pt.free[i]==0&&pt.fixed_value[i]!=0&&!assigned[c.block][c.col]){latent[c.block](c.col)=obs[c.block](c.row)/std::abs(pt.fixed_value[i]);assigned[c.block][c.col]=true;}}
 VectorXd full=VectorXd::Ones(pt.n_free());for(size_t i=0;i<pt.free.size();++i)if(pt.free[i]>0){auto c=rep.cell_for_row[i];if(!c.used)continue;auto& s=obs[c.block];auto& t=latent[c.block];double v=1;switch(c.mat){case model::MatId::Lambda:v=s(c.row)/t(c.col);break;case model::MatId::Theta:v=s(c.row)*s(c.col);break;case model::MatId::Psi:v=t(c.row)*t(c.col);break;case model::MatId::Beta:v=t(c.row)/t(c.col);break;case model::MatId::Nu:v=s(c.row);break;case model::MatId::Alpha:v=t(c.row);break;}full(pt.free[i]-1)=v;}
 MatrixXd scaled=full.cwiseInverse().asDiagonal()*con.K();VectorXd reduced=scaled.colwise().norm().transpose().cwiseInverse();require(reduced.allFinite()&&(reduced.array()>0).all(),"finite coordinate metric");return reduced;
}
