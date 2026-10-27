\name{native_pointer_live}
\alias{native_pointer_live}
\title{Check whether a native pointer has a live address}
\usage{native_pointer_live(pointer)}
\arguments{
\item{pointer}{An external pointer from a prepared or inference object.}
}
\value{A logical scalar: TRUE for an external pointer with a non-null address.}
\description{Inspect process-local pointer state before rebuilding a restored
prepared or inference object.}
\keyword{internal}
