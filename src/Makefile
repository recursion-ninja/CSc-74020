#######
###
#   Environmental Constants
###
#######

SHELL   := /bin/sh
SUBDIRS := curation

#######
###
#   Standard Targets
###
#######

TOPTARGETS := all clean install installdirs

.PHONY: $(TOPTARGETS) $(SUBDIRS)

$(TOPTARGETS): $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)
