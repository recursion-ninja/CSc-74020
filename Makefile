#######
###
#   Environmental Constants
###
#######

SHELL   := /bin/sh
SUBDIRS := tex

#######
###
#   Standard Targets
###
#######

TOPTARGETS := all clean install installdirs pdf

.PHONY: $(TOPTARGETS) $(SUBDIRS)

$(TOPTARGETS): $(SUBDIRS)

$(SUBDIRS):
	$(info Subdir is: $@)
	$(MAKE) -C $@ $(MAKECMDGOALS)
