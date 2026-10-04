// Standalone optional-corpus driver; not linked into the library or CI tests.
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "magmaan/parse/mplus_parser.hpp"
int main(int argc, char** argv) {
  std::string path;
  while(argc == 2 ? (path = argv[1], true) : static_cast<bool>(std::getline(std::cin,path))) {
    std::ifstream file(path); if (!file) return 2;std::ostringstream buffer;buffer<<file.rdbuf();
    const auto source=buffer.str();
    auto reader=magmaan::parse::MplusParser::read(source);
    auto model=magmaan::parse::MplusParser::parse(source);
    const auto rule=[](const auto& result) {
      if(result) return std::string("accepted");
      const auto& detail=result.error().detail;const auto begin=detail.find('['),end=detail.find(']',begin);
      return begin==std::string::npos ? std::string("unclassified") : detail.substr(begin,end-begin+1);
    };
    std::cout<<path<<'\t'<<rule(reader)<<'\t'<<rule(model);
    if (argc == 2) {
      const auto diagnostic = [](const auto& result) {
        auto detail = result ? std::string{} : result.error().detail;
        for (char& c : detail) if (c == '\n' || c == '\r' || c == '\t') c = ' ';
        return detail;
      };
      std::cout << '\t' << diagnostic(reader) << '\t' << diagnostic(model);
    }
    std::cout << '\n';
    if (argc == 2) break;
  }
}
