# MapFly project website

Public URL: <https://GuYan66.github.io/MapFly/>

This static academic project page is deployed from `website/` by
`.github/workflows/pages.yml`. GitHub Pages uses **GitHub Actions** as its source.
No build or package installation is required.

## Local preview

From the repository root:

```bash
python -m http.server 8765 --directory website
```

Open <http://localhost:8765/>.

## Editing

- `index.html`: title, author display, resource buttons, navigation and footer.
- `site-config.js`: public author details, repository/dataset links, citation.
- `static/js/mapfly.js`: sections, scene gallery, videos and PDF reader.
- `static/css/editorial.css`: final typography and responsive layout.
- `assets/`: local paper, figures, videos, icons and fonts.

Page order: cover, abstract, project video, simulation environments, toolchain,
dataset overview, model, navigation examples, paper reader and BibTeX.
The fifteen scenes use a five-column mosaic on desktop and two columns on mobile.
The four demonstrations already run at 10× speed; do not accelerate them again.

The paper is stored at `assets/MapFly.pdf`. Replace that file to update both PDF
links and the embedded reader. Preserve relative asset URLs for project-path hosting.

Anonymous author details are retained from the supplied manuscript. Update these
and the provisional citation only when confirmed public metadata is available.

## License

Based on Eliahu Horwitz's Academic Project Page Template. See
[`TEMPLATE-NOTICE.md`](TEMPLATE-NOTICE.md) for attribution and CC BY-SA 4.0 terms.
The repository's MIT license does not replace the website template license.
Font and icon licenses are included under `assets/`.
