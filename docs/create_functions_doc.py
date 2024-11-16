import importlib
import os
import pkgutil
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import ffit
from ffit.funcs import FitLogic

FONT_FOLDER = Path(__file__).parent / "fonts"
EXPORT_FOLDER = Path(__file__).parent / "functions"


def import_fonts():
    for path in FONT_FOLDER.glob("*"):
        if path.is_file() and (path.suffix == ".ttf" or path.suffix == ".otf"):
            fm.fontManager.addfont(str(path))
            fm.fontManager.addfont(str(path))


MARKDOWN_TEMPLATE = """---
title: {title}
---

### Simples example

```
>>> from ffit.funcs.{filename} import {classname}

# Call the fit method with x and y data.
>>> fit_result = {classname}().fit(x, y)

# The result is a FitResult object that can be unpacked.
>>> res, res_func = fit_result

# One can combine multiple calls in one line.
>>> res = {classname}().fit(x, y, guess=[1, 2, 3, 4]).plot(ax).res
```

<!-- prettier-ignore -->
::: ffit.funcs.{filename}.{classname}

"""

MARKDOWN_PARAM_TEMPLATE = """---
title: {title}
---

<!-- prettier-ignore -->
::: ffit.funcs.{filename}.{classname}
    options:
      show_bases: false
      show_root_heading: false
      summary: false

"""


def get_filename(module):
    return os.path.splitext(os.path.basename(module.__file__ or ""))[0]


def create_plot(cls, filename):
    param_len = len(cls.param.fields())

    param = getattr(cls, "_example_param", np.array([1] * param_len))
    x_min = getattr(cls, "_example_x_min", -1)
    x_max = getattr(cls, "_example_x_max", 1)
    x_points = getattr(cls, "_example_x_points", 100)
    normal_std = getattr(cls, "_example_std", 0.1)
    x = np.linspace(x_min, x_max, x_points)

    y = cls.func(x, *param)
    y += np.random.normal(0, normal_std, len(y))

    with plt.xkcd():
        fig, ax = plt.subplots()
        ax.plot(x, np.real(y), ".", markersize=8)
        res, res_func = cls().fit(x, y)
        ax.plot(x, np.real(res_func(x)))
        ax.set(xlabel="x", ylabel="y", title=f"{cls.__name__} fit")
        fig.tight_layout()

        fig.savefig(EXPORT_FOLDER / f"{filename}_example.png", dpi=300)


def create_markdown(cls, filename):
    with open(EXPORT_FOLDER / f"{filename}.md", "w", encoding="utf-8") as f:
        f.write(
            MARKDOWN_TEMPLATE.format(
                title=cls.__name__, filename=filename, classname=cls.__name__
            )
        )


def create_param_markdown(cls, filename):
    with open(EXPORT_FOLDER / f"{filename}_param.md", "w", encoding="utf-8") as f:
        f.write(
            MARKDOWN_PARAM_TEMPLATE.format(
                title=cls.__name__, filename=filename, classname=cls.__name__
            )
        )


def create_index(names_files: "list[tuple[str, str]]"):
    names_files = sorted(names_files, key=lambda x: x[0])
    text = "# Implemented functions\n\n"
    text += "Here is a list of the all implemented functions sorted alphabetically.\n\n"
    for name, file in names_files:
        text += f"## [{name}]({file}.md)\n\n"
        text += f"[![{file}]({file}_example.png)]({file}.md)\n\n"
    with open(EXPORT_FOLDER / "index.md", "w", encoding="utf-8") as f:
        f.write(text)


def go_through_funcs():
    EXPORT_FOLDER.mkdir(exist_ok=True)
    names_files = []
    # Iterate through all modules in the ffit.funcs package
    for _, module_name, _ in pkgutil.iter_modules(ffit.funcs.__path__):
        module = importlib.import_module(f"ffit.funcs.{module_name}")
        filename = get_filename(module)
        for _, cls in list(vars(module).items()):
            if (
                isinstance(cls, type)
                and issubclass(cls, FitLogic)
                and cls is not FitLogic
            ):
                if getattr(cls, "_doc_ignore", False):
                    continue
                print(filename)
                cls_fit = cls

                create_plot(cls_fit, filename)
                create_markdown(cls_fit, filename)
                names_files.append((cls_fit.__name__, filename))
                create_param_markdown(cls_fit.param, filename)

    create_index(names_files)


if __name__ == "__main__":
    import_fonts()
    go_through_funcs()
