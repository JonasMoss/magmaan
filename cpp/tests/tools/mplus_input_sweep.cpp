// Standalone optional-corpus driver; not linked into the library or CI tests.
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "magmaan/parse/mplus_parser.hpp"
int main() {
  std::string path;
  while(std::getline(std::cin,path)) {
    std::ifstream file(path);std::ostringstream buffer;buffer<<file.rdbuf();
    const auto source=buffer.str();
    auto reader=magmaan::parse::MplusParser::read(source);
    auto model=magmaan::parse::MplusParser::parse(source);
    const auto rule=[](const auto& result) {
      if(result) return std::string("accepted");
      const auto& detail=result.error().detail;const auto begin=detail.find('['),end=detail.find(']',begin);
      return begin==std::string::npos ? std::string("unclassified") : detail.substr(begin,end-begin+1);
    };
    std::cout<<path<<'\t'<<rule(reader)<<'\t'<<rule(model)<<'\n';
  }
}
