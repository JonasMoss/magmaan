# Debug timings exceed about 60 seconds; retain these gates in opt validation.
# This file runs during CTest discovery, after all doctest test includes.
set(slow_test_names
  "parity:Paper examples aggregate fits and ordinal modification indices match lavaan"
  "sim:Pearson Type IV rejection draws match validated quantiles"
  "ordinal:Textbook mixed DWLS: matched setup reproduces lavaan objectives and complete tables"
  "inference:Mixed DWLS policy: exact sampling global law and IJ nested law"
  "inference:DWLS policy IJ covariance agrees with the delete-one jackknife"
  "ordinal:Mixed DWLS IJ covariance agrees with the delete-one jackknife"
  "inference:frontier ML2S MI and releases reduce to complete-data robust tests"
  "ordinal:association ML IJ nonnormal case weights and stratified delete one"
  "inference:DWLS moment nesting: 100-replicate true-null moment diagnostic")
foreach(test_name IN LISTS slow_test_names)
  string(REGEX REPLACE ":.*" "" area "${test_name}")
  set_tests_properties("${test_name}" PROPERTIES LABELS "${area};slow")
endforeach()
