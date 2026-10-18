"""PDE-OBS revision tools (rev8k).

Two contract lines live here:

* ``rev8k_v1_executed`` - the rules the v1 supplementary campaign actually ran with (seed64 of decision D-0001, the
  development split, the mixture rank, ...). They are frozen: :mod:`pdeobs.revision.contracts` reproduces them bit for
  bit and nothing in this package re-seeds an executed result.
* ``rev8k_v2`` - the V2 revision data, observation protocols and numerical reference route
  (:mod:`pdeobs.revision.v2spec`, :mod:`pdeobs.revision.reference`, :mod:`pdeobs.revision.v2data`,
  :mod:`pdeobs.revision.v2pilot`). Every V2 seed is ``seed64("rev8k-v2/<name>", ...)``.

Every public operation is a one-line command, ``python -m pdeobs.revision <command> ...``. Each writes a receipt
naming :data:`REVISION_VERSION` and the contract version, so a dataset or table can be cited by version.
"""
from __future__ import annotations

REVISION_VERSION = "0.3.0"
V2_DATA_CONTRACT = "pdeobs.rev8k.v2.data/1"
V2_REFERENCE_CONTRACT = "pdeobs.rev8k.v2.reference/1"
V1_CONTRACT = "rev8k-contracts-D0001-v1"

__all__ = ["REVISION_VERSION", "V1_CONTRACT", "V2_DATA_CONTRACT", "V2_REFERENCE_CONTRACT"]
