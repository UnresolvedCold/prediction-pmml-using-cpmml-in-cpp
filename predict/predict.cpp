#include "cPMML.h"
#include <iostream>

int main() {
  cpmml::Model model("lr_model.pmml");
  std::unordered_map<std::string, std::string> input1 = {
    {"X", "0"}
  };

  std::unordered_map<std::string, std::string> input2 = {
    {"X", "10"}
  };

  std::cout<<"X = 0 Y = "<<model.predict(input1)<<'\n';
  std::cout<<"X = 10 Y = "<<model.predict(input2)<<'\n';
  
  return 0;
}
