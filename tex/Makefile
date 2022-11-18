#######
###
#   Environmental Constants
###
#######

SHELL := /bin/sh
COMMA := ,
EMPTY :=
SPACE := $(EMPTY) $(EMPTY)
define NEWLINE

$(EMPTY)
endef

#######
###
#   Conditionl Redefinitions
###
#######

extension-figure      ?= png
extension-latex       ?= tex
extension-markdown    ?= md
extension-portabledoc ?= pdf
extension-postscript  ?= ps

#######
###
#   Variables for proposal
###
#######

common-prefix   := CSc-74020-Washburn-Park-Project
filepath-assets := ../data/
filepath-output := ../doc/
row-delimiter   := $(SPACE)\$(NEWLINE)$(SPACE)$(SPACE)$(SPACE)$(SPACE)$(SPACE)$(SPACE)

# Document file paths
proposal-latex  := proposal.$(extension-latex)
proposal-name   := $(common-prefix)-Proposal
proposal-prefix := $(abspath $(filepath-output)$(proposal-name))
proposal-target := $(proposal-prefix).$(extension-portabledoc)

# Build Artifacts
latexmk_accounted_exts := aux fdb_latexmk fls log toc vrb
latexmk_leftover_exts  := $(sort acn bbl glg loc nav snm soc synctex.gz thm)
latexmk_leftover_list  := $(addprefix $(proposal-prefix).,$(latexmk_leftover_exts))

$(info Leftover List:$(NEWLINE)$(latexmk_leftover_list))


#$(subst $(SPACE),$(row-delimiter),\
#$(strip $(foreach ext,(latexmk_leftover_exts),$(proposal-prefix).$(ext))))
#$(subst $(SPACE),$(row-delimiter),$(wildcard $(latexmk_leftover_exts)))

artifact-exts := $(sort \
  $(latexmk_accounted_exts) \
  $(latexmk_leftover_exts) \
  $(extension-portabledoc) \
  )

artifact-list = $(subst $(SPACE),$(row-delimiter),\
$(strip $(foreach ext,$(artifact-exts),$(wildcard $(proposal-prefix).$(ext)))))

#######
###
#   Standard Targets
###
#######

.PHONY: all clean install installdirs pdf

all:: $(proposal-target)

clean:: proposal-clean

install:: $(proposal-target)

installdirs:: $(dir $(proposal-target))

pdf:: $(proposal-target)

#######
###
#   Phony targets
###
#######

.PHONY: proposal proposal-clean

proposal: $(proposal-target)

proposal-clean:
	-rm -f $(artifact-list)
	$(if $(wildcard $(proposal-target)),$(shell rm -f $(proposal-target)))

#######
###
#   Build Targets
###
#######
$(proposal-target): $(proposal-latex)
	latexmk    -jobname=$(basename $(@F)) -output-directory=$(@D) -pdf $<
	latexmk -c -jobname=$(basename $(@F)) -output-directory=$(@D)
	-rm -f $(subst $(SPACE),$(row-delimiter),$(latexmk_leftover_list))

