#include <iostream>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>

#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>


// Inference
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/operations/maximizer.hxx>

int main() {
	typedef opengm::GraphicalModel<double,opengm::Multiplier,opengm::ExplicitFunction<double, size_t, size_t>,opengm::DiscreteSpace<size_t, size_t> > Model;
	typedef Model::IndependentFactorType IndependentFactor;

	size_t n_var=4;
	int n_stats[]={2,2,2,2};

	opengm::DiscreteSpace<size_t,size_t> space;
	for(int i=0;i<n_var;i++)
	{
		space.addVariable(n_stats[i]);   
	}
	Model model(space);

	opengm::ExplicitFunction<double> f1(n_stats,n_stats+2,0);
	opengm::ExplicitFunction<double> f2(n_stats,n_stats+2,0);
	opengm::ExplicitFunction<double> f3(n_stats,n_stats+2,0);
	opengm::ExplicitFunction<double> f4(n_stats,n_stats+2,0);
	f1(0,0)=30;
	f1(0,1)=5;
	f1(1,0)=1;
	f1(1,1)=10;

	f2(0,0)=100;
	f2(0,1)=1;
	f2(1,0)=1;
	f2(1,1)=100;

	f3(0,0)=1;
	f3(0,1)=100;
	f3(1,0)=100;
	f3(1,1)=1;

	f4(0,0)=100;
	f4(0,1)=1;
	f4(1,0)=1;
	f4(1,1)=100;


	size_t vars1[]={0,1};
	size_t vars2[]={1,2};
	size_t vars3[]={2,3};
	size_t vars4[]={0,3};
	Model::FunctionIdentifier fid1=model.addFunction(f1);
	Model::FunctionIdentifier fid2=model.addFunction(f2);
	Model::FunctionIdentifier fid3=model.addFunction(f3);
	Model::FunctionIdentifier fid4=model.addFunction(f4);
	size_t facid1=model.addFactor(fid1,vars1,vars1+2);
	size_t facid2=model.addFactor(fid2,vars2,vars2+2);
	size_t facid3=model.addFactor(fid3,vars3,vars3+2);
	size_t facid4=model.addFactor(fid4,vars4,vars4+2);


	// save model
	opengm::hdf5::save(model, "01_gm.h5", "test");

	typedef opengm::BeliefPropagationUpdateRules<Model, opengm::Integrator> UpdateRules_sum_prod;
	typedef opengm::MessagePassing<Model, opengm::Integrator, UpdateRules_sum_prod, opengm::MaxDistance> BeliefPropagation_sum_prod;

	const size_t maxNumberOfIterations = 100000;
	const double convergenceBound = 1e-90;
	const double damping = 0.0;
	BeliefPropagation_sum_prod::Parameter parameter_s_p(maxNumberOfIterations, convergenceBound, damping); 
	parameter_s_p.inferSequential_ = true;
	// parameter_s_p.useNormalization_=true;

	BeliefPropagation_sum_prod bp_s_p(model, parameter_s_p);


	bp_s_p.infer();

	std::vector<size_t> labeling(4);
	bp_s_p.arg(labeling);

	for (auto &v : labeling) {
		std::cout << v << std::endl;
	}

	IndependentFactor IF;
	double Z=0;
	for(size_t i=0;i<n_var;i++)
	{
		bp_s_p.marginal(i,IF);
		Z=0;
		for(size_t j=0;j<n_stats[i];j++)
		{
			Z=IF(j)+Z;
		}
		std::cout<<"X_"<<i<<": ";
		for(size_t j=0;j<n_stats[i];j++)
		{
			std::cout<<"state: "<<j<<" value: "<<IF(j)/Z<<"; ";
		}
		std::cout<<"\n";
	}

	return 0;

}
