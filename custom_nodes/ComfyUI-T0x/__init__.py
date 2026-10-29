import json
import re

import comfy.utils
import node_helpers
from comfy_api.latest import ComfyExtension, io
from comfy_execution.graph_utils import ExecutionBlocker


class PickByName(io.ComfyNode):
    """For every name in `pick`, the item of `items` whose label in `names` matches it. `items` and `names` are two lists in the
    same order, e.g. a set made row by row from a Prompt Each list and labelled from the same rows."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        t = io.MatchType.Template("item")
        return io.Schema(
            node_id="T0x_PickByName",
            display_name="Pick by Name",
            category="T0x",
            description="Looks items up by name: for every name in 'pick' it outputs the item labelled with that name in 'names'. "
                        "'items' and 'names' must be in the same order. Works with pictures, text or anything else. "
                        "Matching ignores case and surrounding spaces.",
            is_input_list=True,
            inputs=[
                io.MatchType.Input("items", template=t, tooltip="The set to pick from, as a list (pictures may have any size)."),
                io.String.Input("names", force_input=True, tooltip="One label per item, same order as 'items'."),
                io.String.Input("pick", force_input=True, tooltip="The names to look up; one output item per name."),
            ],
            outputs=[io.MatchType.Output(template=t, display_name="items", is_output_list=True)],
        )

    @classmethod
    def execute(cls, items, names, pick) -> io.NodeOutput:
        if len(names) != len(items):
            raise ValueError(f"Pick by Name: {len(items)} items but {len(names)} names; they must pair up one to one.")
        index = {}
        for i, n in enumerate(names):
            index.setdefault(n.strip().lower(), i)
        out = []
        for p in pick:
            i = index.get(p.strip().lower())
            if i is None:
                raise ValueError(f"Pick by Name: nothing named '{p.strip()}'. Available: {', '.join(n.strip() for n in names)}")
            out.append(items[i])
        return io.NodeOutput(out)


class JoinLists(io.ComfyNode):
    """Chains the connected lists into one, in socket order. A socket whose source is muted or missing is skipped, so a layer that
    did not run simply adds nothing."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T0x_JoinLists",
            display_name="Join Lists",
            category="T0x",
            description="Joins several lists into one, in socket order. Sockets with nothing arriving (muted or bypassed layers) are skipped.",
            is_input_list=True,
            inputs=[io.Autogrow.Input("lists", template=io.Autogrow.TemplatePrefix(io.AnyType.Input("list", optional=True), prefix="list", min=0, max=16))],
            outputs=[io.AnyType.Output(display_name="list", is_output_list=True)],
        )

    @classmethod
    def execute(cls, lists) -> io.NodeOutput:
        out = [x for k in sorted(lists, key=lambda k: int(k.removeprefix("list"))) for x in (lists[k] or []) if x is not None]
        if not out:
            raise ValueError("Join Lists: nothing arrived on any socket.")
        return io.NodeOutput(out)


FIELDS = 8


class SplitRow(io.ComfyNode):
    """Splits one row of text like 'O01 | Y000 | a black dress | none' into its fields, trimmed. Missing fields come out empty."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T0x_SplitRow",
            display_name="Split Row",
            category="T0x",
            description="Splits a row of text into fields at the separator (default '|'), trimming spaces. Fields the row does not have come out empty.",
            inputs=[
                io.String.Input("row", force_input=True),
                io.String.Input("separator", default="|"),
            ],
            outputs=[io.String.Output(display_name=f"field {i + 1}") for i in range(FIELDS)],
        )

    @classmethod
    def execute(cls, row, separator) -> io.NodeOutput:
        parts = [p.strip() for p in row.split(separator)]
        return io.NodeOutput(*(parts + [""] * FIELDS)[:FIELDS])


def _parse_list(text):
    """A category box: first line '{tag}', then one value per line as 'code | text', 'code | text | weight' or plain 'text'.
    Empty lines and lines starting with '#' are skipped. A weight repeats the value that many times in the deal."""
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    if not lines or not (lines[0].startswith("{") and lines[0].endswith("}")):
        raise ValueError("Deal Rows: every list box must start with its tag on the first line, e.g. {view}")
    tag, values = lines[0][1:-1].strip(), []
    for n, line in enumerate(lines[1:], 1):
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 1:
            values.append((f"{tag}{n}", parts[0], 1))
        else:
            weight = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1
            values.append((parts[0], parts[1], max(weight, 1)))
    if not values:
        raise ValueError(f"Deal Rows: the {{{tag}}} box has no values")
    return tag, values


class DealRows(io.ComfyNode):
    """Builds one prompt per picture from category boxes. Grid boxes are crossed completely (every combination); every other box is
    dealt across the pictures in a balanced, shuffled order, so each value comes up about equally often (or by its weight) and no
    two values stick together. The same boxes and seed always give the same rows."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T0x_DealRows",
            display_name="Deal Rows",
            category="T0x",
            description="Makes one prompt per picture from category lists. Each list box starts with its {tag}; the template uses the same "
                        "tags. 'grid' names the boxes to combine completely, the rest are dealt evenly across the pictures.",
            inputs=[
                io.Autogrow.Input("lists", template=io.Autogrow.TemplatePrefix(
                    io.String.Input("list", force_input=True, optional=True, tooltip="A category box: {tag} on the first line, then one value per line."),
                    prefix="list", min=1, max=12)),
                io.String.Input("template", multiline=True, default="{view}, {expression}",
                                tooltip="The prompt for every picture. Each {tag} is replaced by that picture's value from the box with the same tag."),
                io.String.Input("grid", default="", tooltip="Tags of the boxes to cross completely, separated by commas, e.g. view, hair. Leave empty to deal everything."),
                io.Int.Input("count", default=0, min=0, max=10000, tooltip="How many pictures. 0 = one pass of the grid."),
                io.Int.Input("seed", default=1, min=0, max=2**31 - 1, tooltip="Changes how the dealt values are shuffled."),
                io.Int.Input("start", default=0, min=0, max=10000, control_after_generate=io.ControlAfterGenerate.fixed, optional=True,
                             tooltip="The grid line the first picture starts on. With count 1 it picks the line; on 'increment' every Run takes the next one."),
            ],
            outputs=[
                io.String.Output(display_name="prompts", is_output_list=True),
                io.String.Output(display_name="names", is_output_list=True),
                io.String.Output(display_name="table"),
            ],
        )

    @classmethod
    def execute(cls, lists, template, grid, count, seed, start=0) -> io.NodeOutput:
        import itertools, random
        boxes = dict(_parse_list(lists[k]) for k in sorted(lists, key=lambda k: int(k.removeprefix("list"))) if lists[k])
        wanted = [t.strip().strip("{}") for t in grid.split(",") if t.strip()]
        for t in wanted:
            if t not in boxes:
                raise ValueError(f"Deal Rows: grid names {{{t}}}, but no box has that tag. Boxes: {', '.join(boxes)}")
        combos = list(itertools.product(*[[v for v in boxes[t] for _ in range(v[2])] for t in wanted])) if wanted else [()]
        count = count or len(combos)
        rows = [dict(zip(wanted, combos[(i + start) % len(combos)])) for i in range(count)]
        for k, (tag, values) in enumerate(boxes.items()):
            if tag in wanted:
                continue
            total = sum(v[2] for v in values)
            share = [count * v[2] / total for v in values]          # largest-remainder split, then a seeded shuffle
            n = [int(s) for s in share]
            for i in sorted(range(len(values)), key=lambda i: share[i] - n[i], reverse=True)[:count - sum(n)]:
                n[i] += 1
            deck = [v for v, c in zip(values, n) for _ in range(c)]
            random.Random(seed * 1000 + k).shuffle(deck)
            for r, v in zip(rows, deck):
                r[tag] = v
        prompts, names, table = [], [], []
        for i, r in enumerate(rows, 1):
            text = template
            for tag, v in r.items():
                text = text.replace("{" + tag + "}", v[1])
            missing = [t for t in boxes if "{" + t + "}" in text]
            if missing or "{" in text:
                raise ValueError(f"Deal Rows: the template has a tag no box fills: {text[text.index('{'):text.index('{') + 30]}")
            codes = [r[t][0] for t in boxes if t in r]
            prompts.append(text); names.append(f"{i:03d}_" + "_".join(codes)); table.append(f"{i:03d}  " + " · ".join(codes))
        return io.NodeOutput(prompts, names, "\n".join(table))


FAMILIES = ("views", "standing", "sitting", "kneeling", "crouching", "lying")


def _route(name, routes):
    """The code in a dealt name ('001_sit-stool' -> 'sit-stool') and what 'routes' sends it to: the line for the code itself,
    else the line for the part before its '-'; None when neither has a line."""
    code = name.split("_", 1)[-1]
    table = {k.strip(): v.strip() for k, v in (line.split("=", 1) for line in routes.splitlines() if "=" in line)}
    return code, table.get(code) or table.get(code.split("-")[0])


class PickFamily(io.ComfyNode):
    """Picks the Body Set's pose box for the picture Body Generation made. The dealt name ('001_sit-stool') gives the code; 'routes'
    sends a code, or the part before its '-', to a family. A family with no box wired stops everything after this node."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T0x_PickFamily",
            display_name="Pick Family",
            category="T0x",
            description="Sends Body Generation's dealt code to the pose box of its family. A family left unwired skips the nodes after it.",
            inputs=[
                io.String.Input("name", force_input=True, tooltip="Body Generation's dealt name, e.g. 001_sit-stool."),
                io.String.Input("routes", multiline=True, default="view = views\nstand = standing\nsit = sitting\nkneel = kneeling\ncrouch = crouching\nlie = lying",
                                tooltip="One line per code: code = family. A code also matches the part before its '-' (sit-stool matches sit)."),
                *[io.String.Input(f, force_input=True, optional=True, tooltip=f"The {f} pose box; leave unwired to skip {f} pictures.") for f in FAMILIES],
            ],
            outputs=[
                io.String.Output(display_name="box"),
                io.String.Output(display_name="code"),
            ],
        )

    @classmethod
    def execute(cls, name, routes, **boxes) -> io.NodeOutput:
        code, family = _route(name, routes)
        if family not in FAMILIES:
            raise ValueError(f"Pick Family: no family for '{code}'. Routes go to: {', '.join(FAMILIES)}")
        if boxes.get(family) is None:
            return io.NodeOutput(ExecutionBlocker(None), ExecutionBlocker(None))
        return io.NodeOutput(boxes[family], code)


class PickSize(io.ComfyNode):
    """Turns the picture landscape or portrait for the dealt line: Body Generation's lying lines get the long side across."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T0x_PickSize",
            display_name="Pick Size",
            category="T0x",
            description="Width and height for the dealt line: a code routed to 'landscape' gets the long side across, 'portrait' the long "
                        "side up, any other code the size as it comes in.",
            inputs=[
                io.String.Input("name", force_input=True, tooltip="Body Generation's dealt name, e.g. 001_lie-back."),
                io.Int.Input("width", force_input=True),
                io.Int.Input("height", force_input=True),
                io.String.Input("routes", multiline=True, default="lie = landscape",
                                tooltip="One line per code: code = landscape or portrait. A code also matches the part before its '-'."),
            ],
            outputs=[io.Int.Output(display_name="width"), io.Int.Output(display_name="height")],
        )

    @classmethod
    def execute(cls, name, width, height, routes) -> io.NodeOutput:
        code, shape = _route(name, routes)
        if shape == "landscape":
            return io.NodeOutput(max(width, height), min(width, height))
        if shape == "portrait":
            return io.NodeOutput(min(width, height), max(width, height))
        if shape is not None:
            raise ValueError(f"Pick Size: '{code}' routes to '{shape}'; use landscape or portrait")
        return io.NodeOutput(width, height)


class PlanAngles(io.ComfyNode):
    """The Angle Set's shot list: from Body Generation's four views (by their dealt codes) and a shots box, one row per shot with
    its source, up to two more references and its prompt. A run with no view in it stops everything after this node."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T0x_PlanAngles",
            display_name="Plan Angles",
            category="T0x",
            description="Lays out the angle shots from Body Generation's views. Shots box: '{shot}' on the first line, then "
                        "'name | source code | reference codes or - | prompt' per line. Pictures are scaled to 'megapixels'.",
            is_input_list=True,
            inputs=[
                io.Image.Input("images", tooltip="Body Generation's pictures (CHECK THE MASTER)."),
                io.String.Input("names", force_input=True, tooltip="Body Generation's dealt names, same order as the pictures."),
                io.String.Input("shots", force_input=True, tooltip="The shots box."),
                io.Float.Input("megapixels", default=1.4, min=0.25, max=8.0, step=0.05),
            ],
            outputs=[
                io.Image.Output(display_name="image1", is_output_list=True),
                io.Image.Output(display_name="image2", is_output_list=True),
                io.Image.Output(display_name="image3", is_output_list=True),
                io.String.Output(display_name="prompt", is_output_list=True),
                io.String.Output(display_name="name", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, images, names, shots, megapixels) -> io.NodeOutput:
        mp = megapixels[0]
        views = {n.split("_", 1)[-1]: img for n, img in zip(names, images) if n.split("_", 1)[-1].startswith("view-")}
        if not views:
            return io.NodeOutput(*[ExecutionBlocker(None)] * 5)      # silent: a run without views just skips the Angle Set

        def fit(img):                                   # ImageScaleToTotalPixels, lanczos, sides on multiples of 32
            h, w = img.shape[1:3]
            s = (mp * 1024 * 1024 / (w * h)) ** 0.5
            nw, nh = max(32, round(w * s / 32) * 32), max(32, round(h * s / 32) * 32)
            return comfy.utils.common_upscale(img.movedim(-1, 1), nw, nh, "lanczos", "disabled").movedim(1, -1)

        scaled = {k: fit(v) for k, v in views.items()}
        lines = [l.strip() for l in shots[0].splitlines() if l.strip() and not l.strip().startswith("#")][1:]
        out = [[], [], [], [], []]
        for line in lines:
            name, source, refs, prompt = [p.strip() for p in line.split("|", 3)]
            refs = [] if refs == "-" else [r.strip() for r in refs.split(",")]
            missing = [c for c in [source] + refs if c not in scaled]
            if missing:
                raise ValueError(f"Plan Angles: shot '{name}' needs {', '.join(missing)}, but the run made {', '.join(scaled)}")
            pics = [scaled[c] for c in [source] + refs] + [None, None]
            for o, v in zip(out, pics[:3] + [prompt, name]):
                o.append(v)
        return io.NodeOutput(*out)


class ReferenceList(io.ComfyNode):
    """Set Reference Multi Latent for a list: every picture of a list (e.g. the picks of Load Images from Folder Pixaroma) becomes
    its own reference, in list order, so the prompt can cite them as <ref_image_0>, <ref_image_1>, ..."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T0x_ReferenceList",
            display_name="Reference List",
            category="T0x",
            description="Adds every picture of a list to the conditioning as its own reference, in list order (the first is "
                        "<ref_image_0>). Like Set Reference Multi Latent, but for any number of pictures from one list.",
            is_input_list=True,
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative", optional=True),
                io.Vae.Input("vae"),
                io.Image.Input("images", tooltip="The reference pictures as a list; they may have different sizes."),
                io.Float.Input("megapixels", default=1.0, min=0.0, max=16.0, step=0.01,
                               tooltip="Size of each reference, aspect ratio kept. 0 keeps every picture's own size."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative", tooltip="The positive output when no negative is connected."),
            ],
        )

    @classmethod
    def execute(cls, positive, vae, images, megapixels, negative=None) -> io.NodeOutput:
        vae, mp = vae[0], megapixels[0]
        refs = []
        for img in images:
            if mp > 0:
                h, w = img.shape[1:3]
                s = (mp * 1024 * 1024 / (w * h)) ** 0.5
                img = comfy.utils.common_upscale(img.movedim(-1, 1), max(1, round(w * s)), max(1, round(h * s)), "lanczos", "disabled").movedim(1, -1)
            refs.append(vae.encode(img))
        positive = node_helpers.conditioning_set_values(positive[0], {"reference_latents": refs}, append=True)
        negative = positive if negative is None else node_helpers.conditioning_set_values(negative[0], {"reference_latents": refs}, append=True)
        return io.NodeOutput(positive, negative)


PIECES = {  # kind: (order down the body, share of a column's height)
    "hat": (0, 0.5), "outerwear": (1, 1.3), "top": (2, 1.0), "dress": (3, 2.2), "undergarments": (4, 0.8),
    "bottoms": (5, 1.7), "socks": (6, 0.6), "shoes": (7, 0.7), "accessory": (8, 0.5)}
KIND_WORDS = {  # what the upsampler writes instead of a kind
    "hat": "cap beanie headwear", "outerwear": "jacket coat blazer cardigan", "top": "shirt t-shirt tee blouse sweater hoodie tank",
    "dress": "gown jumpsuit one-piece romper", "undergarments": "lingerie bra underwear panties briefs bodysuit",
    "bottoms": "pants trousers jeans skirt shorts leggings", "socks": "tights stockings legwear",
    "shoes": "footwear boots sneakers heels sandals", "accessory": "bag handbag jewelry jewellery sunglasses glasses belt necklace earrings"}
KINDS = {w: k for k, words in KIND_WORDS.items() for w in words.split()} | {k: k for k in PIECES}
PIECE_LINE = re.compile(r"(?:image\s*)?(\d+)\s*\|\s*([^|\n]+?)\s*\|\s*(.+?)(?=\s*(?:image\s*)?\d+\s*\||\Z)", re.I | re.S)
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
CATALOGUE_LEAD = ("A clean e-commerce product catalogue on a pure white background, every piece laid flat, fully visible and shown exactly "
                  "as in its picture, with no text, labels, captions or other objects anywhere.")


class CatalogueBoxes(io.ComfyNode):
    """One catalogue box per piece the upsampler named, laid out down the body in two columns, plus the prompt that ties each box to
    its picture. Feeds Create Bounding Boxes' 'bboxes' input (the canvas redraws from it) and FLUX 3 Image Layout Prompt."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T0x_CatalogueBoxes",
            display_name="Catalogue Boxes",
            category="T0x",
            is_input_list=True,
            description="Reads the pieces as 'image number | kind | description' lines and makes one box per piece: hats and outerwear "
                        "at the top, shoes at the bottom, in two columns, each box as tall as its kind needs. Kinds: "
                        + ", ".join(PIECES) + "; anything else counts as accessory.",
            inputs=[
                io.String.Input("pieces", force_input=True, tooltip="One line per piece: image number | kind | description."),
                io.Int.Input("width", default=1024, min=256, max=4096, step=16, tooltip="Same as Create Bounding Boxes' width."),
                io.Int.Input("height", default=1536, min=256, max=4096, step=16, tooltip="Same as Create Bounding Boxes' height."),
                io.String.Input("lead", multiline=True, default=CATALOGUE_LEAD, tooltip="The prompt's first sentence; one sentence per box follows."),
                io.Image.Input("images", optional=True, tooltip="The pictures the pieces come from; with only one, every piece is taken from it."),
            ],
            outputs=[
                io.String.Output(display_name="bboxes", tooltip="Into Create Bounding Boxes' 'bboxes' input."),
                io.String.Output(display_name="prompt", tooltip="Into FLUX 3 Image Layout Prompt's 'prompt'."),
            ],
        )

    @classmethod
    def execute(cls, pieces, width, height, lead, images=None) -> io.NodeOutput:
        pieces, width, height, lead = pieces[0], width[0], height[0], lead[0]
        found = []                                      # one piece per 'n | kind | text', on its own line or run together
        for m in PIECE_LINE.finditer(pieces):
            n = 1 if images is not None and len(images) == 1 else int(m[1])
            found.append((n, KINDS.get(m[2].strip().lower(), "accessory"), " ".join(m[3].split()).rstrip(".")))
        if not found:
            raise ValueError(f"Catalogue Boxes: no 'image number | kind | description' line in: {pieces[:300]}")
        found.sort(key=lambda p: PIECES[p[1]][0])
        cols, load = ([found] if len(found) == 1 else [[], []]), [0.0, 0.0]
        if len(found) > 1:
            for p in found:
                c = 0 if load[0] <= load[1] else 1
                cols[c].append(p); load[c] += PIECES[p[1]][1]
        margin, gap = 40, 24
        cw = (width - 2 * margin - gap * (len(cols) - 1)) / len(cols)
        boxes, lines = [], []
        for c, col in enumerate(cols):
            y, free = margin, height - 2 * margin - gap * (len(col) - 1)
            total = sum(PIECES[p[1]][1] for p in col)
            for n, kind, desc in col:
                h = free * PIECES[kind][1] / total
                boxes.append({"x": round(margin + c * (cw + gap)), "y": round(y), "width": round(cw), "height": round(h),
                              "metadata": {"type": "obj", "text": "", "desc": f"{kind}: {desc}", "palette": []}})
                image = ORDINALS[n - 1] if 0 < n <= len(ORDINALS) else f"number {n}"
                piece = desc if desc.lower().startswith(("a ", "an ", "the ")) else "the " + desc
                lines.append(f"<obj_{len(boxes)}> holds {piece} from the {image} image.")
                y += h + gap
        return io.NodeOutput(json.dumps(boxes), " ".join([lead.strip()] + lines))


PICK_ROUTES = """# first matching line wins; patterns are matched against the picture's name, * = anything
# the reference views are named by where the nose points in the frame (profleft = nose to the left edge)
*view-back* = skip
*back-* = skip
*backleft-* = skip
*backright-* = skip
*view-profr* = profleft
*view-profl* = profright
*frontright-* = 34left
*frontleft-* = 34right
*right-* = profleft
*left-* = profright
* = front"""


class PickReference(io.ComfyNode):
    """The Face Pass reference for one picture, found from its dealt name: 'routes' turns the name into a view (or skip), and an
    expression code anywhere in the name picks that expression when the bank has it for the view, else the view's neutral face.
    The bank is a folder of '<view>_<expression>.png' files under ComfyUI's input folder, or, when bank_images is wired, the Face
    Set's pictures with their dealt names ('001_front_smile_h1_...': the first two words after the number are view and expression;
    the first picture of each pair is used)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T0x_PickReference",
            display_name="Pick Reference",
            category="T0x",
            description="Finds the face reference that matches a picture from its dealt name (e.g. frontright-elev, 001_street_smile). "
                        "Routes give the view, an expression code in the name gives the expression; the bank holds <view>_<expression>.png. "
                        "A route to 'skip' (back views) sets 'skip', so a switch can keep the original picture.",
            inputs=[
                io.String.Input("name", force_input=True, tooltip="The picture's dealt name or file name."),
                io.String.Input("bank", default="faces/bank", tooltip="Folder under ComfyUI's input folder with <view>_<expression>.png files."),
                io.String.Input("routes", multiline=True, default=PICK_ROUTES,
                                tooltip="One line per rule: pattern = view (or skip). First match wins; * matches anything."),
                io.String.Input("expressions", force_input=True, optional=True,
                                tooltip="Optional EXPRESSIONS box (Deal Rows format): 'expression' then comes out as that line's text, not the code."),
                io.Image.Input("bank_images", optional=True, tooltip="Optional: the Face Set's pictures (list), used as the bank instead of the folder."),
                io.String.Input("bank_names", force_input=True, optional=True, tooltip="The Face Set's dealt names, same order as bank_images."),
            ],
            outputs=[
                io.Image.Output(display_name="reference", is_output_list=True),
                io.String.Output(display_name="view", is_output_list=True),
                io.String.Output(display_name="expression", is_output_list=True),
                io.Boolean.Output(display_name="skip", is_output_list=True),
            ],
            is_input_list=True,
        )

    @classmethod
    def execute(cls, name, bank, routes, expressions=None, bank_images=None, bank_names=None) -> io.NodeOutput:
        import fnmatch, logging, os
        import numpy as np, torch
        from PIL import Image
        import folder_paths
        bank, routes, expressions = bank[0], routes[0], expressions[0] if expressions else None
        if bank_images:
            if not bank_names or len(bank_names) != len(bank_images):
                raise ValueError("Pick Reference: bank_names must be wired with bank_images, one name per picture.")
            pictures = {}
            for img, n in zip(bank_images, bank_names):
                words = [t for t in re.split(r"[_\s]+", os.path.basename(n).lower()) if t and not t.isdigit()]
                if len(words) >= 2: pictures.setdefault(f"{words[0]}_{words[1]}", img[:1])
            files, source = set(pictures), "the Face Set"
        else:
            folder = os.path.join(folder_paths.get_input_directory(), bank)
            if not os.path.isdir(folder):
                raise ValueError(f"Pick Reference: no bank folder '{bank}' in the input folder.")
            files, source = {os.path.splitext(f)[0].lower() for f in os.listdir(folder) if f.lower().endswith(".png")}, bank
        labels = dict((c, t) for c, t, _ in _parse_list(expressions)[1]) if expressions else {}
        out = ([], [], [], [])
        for one in name:
            key = os.path.basename(one).lower()
            view = None
            for line in routes.splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    pat, v = (s.strip() for s in line.split("=", 1))
                    if fnmatch.fnmatch(key, pat.lower()):
                        view = v.lower(); break
            if view is None:
                raise ValueError(f"Pick Reference: no route matches '{one}'. Add a line, e.g. '* = front'.")
            skip = view == "skip"                              # a back view: still runs (blocking one item of a list would also
            if skip: view = "front"                            # block it at the switch); 'skip' makes the switch keep the original
            expression = next((t for t in re.split(r"[_\s]+", key) if f"{view}_{t}" in files and t != "neutral"), "neutral")
            stem = f"{view}_{expression}"
            if stem not in files:                          # a bank without the pair: the view's or the front's neutral, then any
                stem = next((s for s in (f"{view}_neutral", "front_neutral") if s in files), None) or next(
                    (s for v in (view, "front") for s in sorted(files) if s.startswith(v + "_")), min(files, default=None))
                if stem is None:
                    raise ValueError(f"Pick Reference: {source} is empty (route for '{one}').")
                logging.info(f"Pick Reference: no '{view}_{expression}' in {source}, '{one}' takes '{stem}'")
                expression = stem.split("_", 1)[1]
            if bank_images:
                ref = pictures[stem]
            else:
                img = Image.open(os.path.join(folder, stem + ".png")).convert("RGB")
                ref = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)[None]
            for o, v in zip(out, (ref, "skip" if skip else view, labels.get(expression, expression), skip)): o.append(v)
        return io.NodeOutput(*out)


def _holds_model(value):
    import comfy.model_patcher
    if isinstance(value, (list, tuple)):
        return any(_holds_model(v) for v in value)
    if isinstance(value, dict):
        return any(_holds_model(v) for v in value.values())
    return comfy.model_patcher.is_model_patcher_output(value)


class ReleaseModels(io.ComfyNode):
    """Frees every model the run no longer needs, in the middle of a run: ComfyUI's RAM cache keeps the current run's models
    (Dynamic VRAM) until the queue item ends. Pictures and names pass through untouched; wiring them through this node is what
    makes it run after the layers that feed it and before the layers it feeds."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        pairs = [("faces", "Face Set"), ("angles", "Angle Set"), ("body", "Body Set")]
        inputs, outputs = [], []
        for key, label in pairs:
            inputs += [io.Image.Input(f"{key}_images", optional=True, tooltip=f"{label} pictures (list)."),
                       io.String.Input(f"{key}_names", optional=True, force_input=True, tooltip=f"{label} names, same order.")]
            outputs += [io.Image.Output(display_name=f"{key}_images", is_output_list=True),
                        io.String.Output(display_name=f"{key}_names", is_output_list=True)]
        return io.Schema(
            node_id="T0x_ReleaseModels",
            display_name="Release Models",
            category="T0x",
            description="Waits for everything wired into it, then drops every cached model, CLIP and VAE from RAM and VRAM and "
                        "passes the pictures and names through. A model a later node still needs stays (or reloads from disk).",
            is_input_list=True,
            inputs=inputs,
            outputs=outputs,
        )

    @classmethod
    def execute(cls, **kw) -> io.NodeOutput:
        import gc, logging
        import psutil
        import comfy.model_management as mm
        before = psutil.virtual_memory().available
        dropped = 0
        executor = next((o for o in gc.get_objects() if type(o).__name__ == "PromptExecutor" and hasattr(o, "caches")), None)
        if executor is None:
            logging.warning("T0x Release Models: ComfyUI's executor not found; only unloading VRAM")
        else:
            cache = executor.caches.outputs
            for key in [k for k, e in list(cache.cache.items()) if e is not None and _holds_model(e.outputs)]:
                cache.cache.pop(key, None)
                for side in ("used_generation", "timestamps", "children"):
                    getattr(cache, side, {}).pop(key, None)
                dropped += 1
        mm.unload_all_models()
        mm.cleanup_models()
        gc.collect()
        mm.soft_empty_cache(True)
        after = psutil.virtual_memory().available
        logging.info(f"T0x Release Models: dropped {dropped} cached model results, free RAM {before / 2**30:.1f} -> {after / 2**30:.1f} GB")
        return io.NodeOutput(*[kw.get(f"{k}_{s}") or [] for k in ("faces", "angles", "body") for s in ("images", "names")])


class Choose(io.ComfyNode):
    """A switch whose choice is a widget, so APP mode can expose it (the Pixaroma Switch keeps its choice in hidden node state).
    Only the chosen input is evaluated."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T0x_Choose",
            display_name="Choose",
            category="T0x",
            description="Passes on the input numbered 'choice'; the other inputs are never run. 'choice' also comes out as a number, "
                        "so other nodes can follow the same switch.",
            inputs=[io.Int.Input("choice", default=1, min=1, max=12, tooltip="The number of the input to pass on.")] +
                   [io.AnyType.Input(f"input_{i}", optional=True, lazy=True) for i in range(1, 13)],
            outputs=[io.AnyType.Output(display_name="output"), io.Int.Output(display_name="choice")],
        )

    @classmethod
    def check_lazy_status(cls, choice, **inputs):
        key = f"input_{choice}"
        return [key] if key in inputs and inputs[key] is None else []

    @classmethod
    def execute(cls, choice, **inputs) -> io.NodeOutput:
        value = inputs.get(f"input_{choice}")
        if value is None:
            raise ValueError(f"Choose: input {choice} is not connected")
        return io.NodeOutput(value, choice)


def _unread(value):
    """A lazy list input ComfyUI has not run yet arrives as (None,)."""
    return value is None or (len(value) == 1 and value[0] is None)


class ForFamily(io.ComfyNode):
    """Passes on the Views pair or the poses pair of a list (pictures + names), by FAMILY's number: 1 = Views, anything else = a
    pose family. Only the chosen pair is run, so the other family's layers, which switch themselves off, never reach what follows.
    A side left unwired skips everything after this node for that family."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        pairs = [("views", "Views (FAMILY 1)"), ("poses", "pose families (FAMILY 2 and up)")]
        inputs = [io.Int.Input("family", force_input=True, tooltip="FAMILY's number: the 'choice' output of the FAMILY switch.")]
        for side, label in pairs:
            inputs += [io.Image.Input(f"{side}_images", optional=True, lazy=True, tooltip=f"The pictures for {label}."),
                       io.String.Input(f"{side}_names", optional=True, lazy=True, force_input=True, tooltip=f"Their names, same order.")]
        return io.Schema(
            node_id="T0x_ForFamily",
            display_name="For Family",
            category="T0x",
            description="Picks the Views pair (FAMILY 1) or the poses pair (any other family) of pictures and names; only that pair "
                        "is run. An unwired pair skips what follows for that family.",
            is_input_list=True,
            inputs=inputs,
            outputs=[io.Image.Output(display_name="images", is_output_list=True),
                     io.String.Output(display_name="names", is_output_list=True)],
        )

    @classmethod
    def _side(cls, family):
        return "views" if family[0] == 1 else "poses"

    @classmethod
    def check_lazy_status(cls, family, **pairs):
        side = cls._side(family)
        return [k for k in (f"{side}_images", f"{side}_names") if k in pairs and _unread(pairs[k])]

    @classmethod
    def execute(cls, family, **pairs) -> io.NodeOutput:
        side = cls._side(family)
        images, names = pairs.get(f"{side}_images"), pairs.get(f"{side}_names")
        if images is None:
            return io.NodeOutput(ExecutionBlocker(None), ExecutionBlocker(None))
        if names is not None and len(names) != len(images):
            raise ValueError(f"For Family: {len(images)} {side} pictures but {len(names)} names.")
        return io.NodeOutput(images, names if names is not None else [])


class T0xExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PickByName, JoinLists, SplitRow, DealRows, PickFamily, PickSize, PlanAngles, ReferenceList, CatalogueBoxes, PickReference,
                ReleaseModels, Choose, ForFamily]


async def comfy_entrypoint() -> T0xExtension:
    return T0xExtension()
