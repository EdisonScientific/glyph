# Vendored public scorer checkouts

The rescoring pipeline imports DS4SD's scorer without modifying it.

- `vendor/MarkushGrapher`: `35c71a0b4423983920b4703f52f46c5bca9ae631`
- `vendor/MarkushGenerator`: `17580c20d266886ef9ae22a50aab51444e0898b2`
- MG2 inference transformers fork (`lucas-morin/transformers`):
  `867ac6e128e2e71ecc12a9c1b65e48e451245a67`
- MG2 inference MolScribe fork (`lucas-morin/MolScribe`):
  `b15c96e10da5e2ec4444a877d1031d8d10190c4d`

The last two checkouts are created under the ignored local reproduction root by
`reproduce_mg2_baseline.sh`; they are listed here because upstream's setup script
otherwise follows their floating default branches.

These are normal git clones in this working tree, not submodules. If recreating
from a fresh checkout:

```bash
git clone https://github.com/DS4SD/MarkushGrapher.git tools/mg2_scorer/vendor/MarkushGrapher
git -C tools/mg2_scorer/vendor/MarkushGrapher checkout 35c71a0b4423983920b4703f52f46c5bca9ae631

git clone https://github.com/DS4SD/MarkushGenerator.git tools/mg2_scorer/vendor/MarkushGenerator
git -C tools/mg2_scorer/vendor/MarkushGenerator checkout 17580c20d266886ef9ae22a50aab51444e0898b2
```
