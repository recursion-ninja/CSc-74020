#######
###
#   Environmental Constants
###
#######

SHELL   := /bin/sh
SUBDIRS := src tex

filepath-binary := $(abspath bin)/
curation-byname := curate-dataset
curation-binary := $(filepath-binary)curate-json
curation-script := $(filepath-binary)$(curation-byname).sh

#######
###
#   Standard Targets
###
#######

TOPTARGETS := all clean install installdirs pdf

.PHONY: $(TOPTARGETS) $(SUBDIRS)

$(TOPTARGETS): $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

$(curation-byname): $(curation-binary) $(curation-script)
	$(curation-script)
