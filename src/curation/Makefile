SHELL=/bin/bash

#######
###
#   Variables for 'Curation'
###
#######

filepath-output := $(abspath ../../bin)/
curation-binary := curate-json
curation-target := $(filepath-output)$(curation-binary)

#######
###
#   Standard Targets
###
#######

.PHONY: all clean install installdirs

all:: $(curation-target)

clean::
	-rm -f $(curation-target)
	cabal clean
	cabal update

install:: $(curation-target)

installdirs:: $(dir $(curation-target))

#######
###
#   Build Targets
###
#######

$(curation-target): $(dir $(filepath-target)) cabal.project curate-json.cabal app/*
	cabal install exe:$(@F) \
	    --installdir=$(@D) \
	    --install-method=copy \
	    --overwrite-policy=always \
	    --enable-profiling \
	    --enable-executable-profiling

$(dir $(curation-target)):
	mkdir -p $@

