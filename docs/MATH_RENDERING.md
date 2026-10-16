# Readable formulas and editable manuscript sources

Both manuscript-template pages display their mathematics as self-contained SVG images. Matrix rows, norm bars, Greek symbols, fractions and subscripts are embedded as vector paths, without depending on GitHub's mathematical macro policy or browser fonts.

- [Original revision reading view](../research_output/revision/manuscript_template.md)
- [Integrated revision reading view](../research_output/revision/sci_revision/manuscript_template.md)
- [Original revision LaTeX source](../research_output/revision/manuscript_template.md.in)
- [Integrated revision LaTeX source](../research_output/revision/sci_revision/manuscript_template.md.in)

These are working templates. Tokens such as `[AUTHOR]`, `{{revised_abstract}}`, `[TABLE:...]` and `[FIG:...]` are intentional build markers. Filled results, figures and the final Word/PDF manuscript are supplied in the complete DOI archive linked from the README. No missing scientific results have been invented for these reading views.

The two `.md.in` files preserve the original template bytes. The current Git authoring and equation-verification scripts read those sources, keeping the native mathematical expressions available for Word conversion. The adjacent `.md` files are generated for reading on GitHub. The frozen DOI archive retains its original filenames and checksums.

## Rebuild the reading views

With Node.js 20 or later and pnpm, run from the repository root:

```sh
pnpm install --dir tools/math-renderer --frozen-lockfile --ignore-scripts
node scripts/render_math.mjs
node scripts/render_math.mjs --check
```

The equation [manifest](assets/math/manifest.json) records all 104 formula occurrences, their source expressions, and their generated SVG checksums. Shared formulas use the same image. The build rejects invalid TeX, oversized equations, external fonts/scripts, or stale generated output. GitHub Actions checks this presentation independently of the scientific training pipeline.

MathJax 3.2.2 supplies the renderer and embedded glyph outlines under the [Apache 2.0 license](../tools/math-renderer/MATHJAX-LICENSE.txt). The SVGs have high-contrast white backgrounds, vector scaling, and LaTeX text alternatives. Changes here concern presentation and source routing, not model protocols, fitted states, or reported scientific results.
