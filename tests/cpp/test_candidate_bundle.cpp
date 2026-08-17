#include "weaveode/candidate_bundle.hpp"
#include <cassert>
int main(){weaveode::CandidateBundle bundle(5,3);assert(bundle.count()==5);assert(bundle.dimension()==3);assert(bundle.states().size()==15);bundle.branch_id()[2]=7;bundle.component_id()[2]=4;bundle.path_id()[2]=11;assert(bundle.branch_id()[2]==7);return 0;}
