// Optional corpus lowering driver, built by check_mplus_sanitizers.py, never CI.
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "magmaan/api/sem.hpp"

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream file(argv[1]);
  if (!file) return 2;
  std::ostringstream buffer; buffer << file.rdbuf();
  auto result = magmaan::api::model_from_mplus(buffer.str());
  if (result) std::cout << "accepted\t\n";
  else {
    auto detail = result.error().detail;
    for (char& c : detail) if (c == '\n' || c == '\r' || c == '\t') c = ' ';
    const auto begin = detail.find('['), end = detail.find(']', begin);
    std::cout << (begin == std::string::npos ? "unclassified" : detail.substr(begin, end-begin+1))
              << '\t' << detail << '\n';
  }
}
