#include <iostream>

#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>


// Inference
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/operations/maximizer.hxx>

int main(int argc, char** argv) {
   //*******************
   //** Typedefs
   //*******************
   typedef int ValueType;             // type used for values
   typedef size_t IndexType;          // type used for indexing nodes and factors (default : size_t)
   typedef size_t LabelType;          // type used for labels (default : size_t)
   typedef opengm::Multiplier OpType;      // operation used to combine terms

   // shortcut for explicite function
   typedef opengm::ExplicitFunction<ValueType,IndexType,LabelType> ExplicitFunction;
   // list of all function the model cal use (this trick avoids virtual methods) - here only one
   typedef opengm::meta::TypeListGenerator<ExplicitFunction>::type FunctionTypeList;   
   // type used to define the feasible statespace
   typedef opengm::DiscreteSpace<IndexType, LabelType> Space;          
	
   // type of the model
   typedef opengm::GraphicalModel<ValueType,OpType,FunctionTypeList,Space> Model;                 
   // type of the function identifier
   typedef Model::FunctionIdentifier FunctionIdentifier;

   // inference type definitions
   typedef opengm::Maximizer InferOpType;
   typedef opengm::BeliefPropagationUpdateRules<Model, InferOpType> UpdateRules;
   typedef opengm::MessagePassing<Model, InferOpType, UpdateRules, opengm::MaxDistance> BeliefPropagation;


   //*******************
   //** Code
   //*******************
   std::cout << "Start building the model ... "<<std::endl;
   // Build empty Model, we can also use SimpleDiscreteSpace
   LabelType numbersOfLabels[] = {2, 2, 2, 2};
   Model gm(Space(numbersOfLabels, numbersOfLabels + 4));

   // add 2nd order function and factors to the model
   {
	   IndexType links[4][2] = {{0,1},
		   {1,2}, 
		   {2,3}, 
		   {0,3}}; // {3,0} cause OpenGM Error: variable indices of a factor must be sorted
	   ValueType phiValues[4][2][2] = {
		   {{30, 5}, {1,10}},
		   {{100,1}, {1,100}},
		   {{1,100}, {100,1}},
		   {{100,1}, {1,100}}
	   };

	   LabelType shape[] = {2, 2};
	   LabelType state[] = {0,0};
	   size_t linkIndex = 0;
	   for (auto &vars : links) {
		   std::cout << "(" << vars[0] << "," << vars[1] << ")" << std::endl;
		   std::cout << "link potential:" << std::endl;
		   ExplicitFunction f(shape, shape + 2);
		   for (state[0] = 0; state[0] < gm.numberOfLabels(vars[0]); state[0]++) {
			   for (state[1] = 0; state[1] < gm.numberOfLabels(vars[1]); state[1]++) {
				   std::cout << phiValues[linkIndex][state[0]][state[1]] << "\t";
				   f(state) = phiValues[linkIndex][state[0]][state[1]];
			   }
			   std::cout << std::endl;
		   }
		   FunctionIdentifier fid = gm.addFunction(f);
		   gm.addFactor(fid, vars, vars + 2);
		   linkIndex++;
		   //
	   }
   }

   // View some model information
   std::cout << "The model has " << gm.numberOfVariables() << " variables."<<std::endl;
   for(size_t i=0; i<gm.numberOfVariables(); ++i){
      std::cout << " * Variable " << i << " has "<< gm.numberOfLabels(i) << " labels."<<std::endl; 
   } 
   std::cout << "The model has " << gm.numberOfFactors() << " factors."<<std::endl;
   for(size_t f=0; f<gm.numberOfFactors(); ++f){
      std::cout << " * Factor " << f << " has order "<< gm[f].numberOfVariables() << "."<<std::endl; 
   }

   LabelType label0[] = {0,1,1,0};
   LabelType label1[] = {0,0,0,1};
   std::cout << "The Labeling (" <<label0[0]<<","<<label0[1]<<","<<label0[2]<<","<<label0[3]<<")  has the energy "<<gm.evaluate(label0)<<"."<<std::endl;
   std::cout << "The Labeling (" <<label1[0]<<","<<label1[1]<<","<<label1[2]<<","<<label1[3]<<")  has the energy "<<gm.evaluate(label1)<<"."<<std::endl;


   // **************
   // *** Inference
   // **************
   const size_t maxNumberOfIterations = 100;
   const double convergenceBound = 1e-7;
   const double damping = 0.9;
   BeliefPropagation::Parameter parameter(maxNumberOfIterations, convergenceBound, damping);
   BeliefPropagation bp(gm, parameter);

   // optimize (approximately)
   BeliefPropagation::VerboseVisitorType visitor;
   bp.infer(visitor);
   // bp.infer(); // non verbose mode
   // obtain the (approximate) argmax
   std::vector<size_t> labeling(4);
   bp.arg(labeling);

   for (auto &v : labeling) {
	   std::cout << v << std::endl;
   }
   
   return 0;
}
