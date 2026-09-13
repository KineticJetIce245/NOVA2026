# Incorporated UI provenance

The supplied UI source and its complete backend/frontend tests were incorporated
from https://github.com/yut31/attune-ui, local commit c331022 (including the NOVA
integration developed for this task). All runtime sources now live in this
NOVA2026 checkout under backend/ and frontend/.

There are no Git submodules, linked working trees, symlinks to sibling repos, or
runtime downloads of application source. The default launcher and real-data
end-to-end test use only this checkout. Dependencies install from their manifests;
participant recordings remain local-only as required by the original handoff.

Only NOVA2026/audio_flo is a publishing destination. The temporary local commits
in the sibling repositories were not pushed and are unnecessary for this branch.
