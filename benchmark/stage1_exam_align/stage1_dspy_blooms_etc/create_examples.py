#!/usr/bin/env python3
"""create_examples.py in src/microchat.

This script generates vector graphics for multiple-choice questions from a CSV file.
"""

import cairo


import PIL.Image
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Any

import click
import pandas as pd
from dotenv import find_dotenv
from dotenv import load_dotenv
from loguru import logger

from microchat import LOG_DIR


def hex_to_rgb(hex_color: str, opacity: float = 1.0) -> tuple:
    """Convert hex color to an RGBA tuple."""
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return (*rgb, opacity)


def wrap_text(context: cairo.Context, text: str, max_width: int):
    """Wrap text based on the maximum width and the font settings of the context."""

    output_lines = []
    lines = text.split("\n")
    for elem in lines:
        words = elem.split()
        line = ""
        for word in words:
            if word in ["\n"]:
                output_lines.append(line)
                line = ""
                continue

            test_line = f"{line} {word}".strip()
            xbearing, ybearing, width, height, xadvance, yadvance = (
                context.text_extents(test_line)
            )

            if width <= max_width:
                line = test_line
            else:
                output_lines.append(line)
                line = word
        if line:
            output_lines.append(line)

    return output_lines


def create_mcq_graphics(
    question: str,
    options: list,
    correct_index: int,
    explanation: str,
    output_path: Path,
    research_subject: str = None,
    bloom_level: str = None,
    question_color="#000000",
    option_color="#000000",
    background_color="#FFFFFF",
    answer_background="#DFFFD6",
    explanation_color="#333333",
    subject_color="#3366CC",
    blooms_color="#228B22",
    specimen_color="#8B4513",
    error_category_color="#B22222",
    opacity=1.0,
    line_spacing=3,
    width=650,
    base_height=200,
    font_size=20,
    section_spacing=25,
    question_prefix: Optional[str] = None,
    explanation_prefix: Optional[str] = "AI",
    prediction: Optional[int] = None,
    error_category: Optional[str] = None,
    error_rationale: Optional[str] = None,
    specimen: Optional[str] = None,
    image_filepath: Optional[Path] = None,
    dry_run=False,
):
    """Create a graphic representation of an MCQ with customizable colors and text wrapping."""
    # Initial dynamic height setup based on content
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, base_height)
    context = cairo.Context(surface)
    context.set_font_size(font_size)

    margin_x, margin_y = 30, 40
    y_offset = margin_y

    # Pre-calculate the height needed for metadata, question, options, and explanation
    metadata_height = 0
    if research_subject or bloom_level or error_category or specimen:
        metadata_height = (
            font_size + line_spacing + 40
        )  # Extra padding for metadata at the end

    # Question prefix
    if question_prefix:
        question = f"{question_prefix}: {question}"

    question_lines = wrap_text(context, question, width - 2 * margin_x)

    # filter options to correct_index and prediction
    options = [options[correct_index], options[prediction]] if prediction else options
    option_lines_wrapped = [
        wrap_text(context, option, width - 2 * margin_x - 5) for option in options
    ]

    # Explanation prefix
    if explanation_prefix:
        explanation = f"{explanation_prefix}: {explanation}"

    explanation_lines = wrap_text(context, f"{explanation}", width - 2 * margin_x)

    if isinstance(error_rationale, str):
        error_rationale = wrap_text(
            context, f"Error reason: {error_rationale}", width - 2 * margin_x - 5
        )

    # Dynamic height calculation
    question_height = len(question_lines) * (font_size + line_spacing)
    line_height = font_size - 5  # 10
    options_height = sum(
        len(lines) * (font_size + line_spacing) + line_height
        for lines in option_lines_wrapped
    )
    explanation_height = len(explanation_lines) * (font_size + line_spacing)
    error_rationale_height = (
        len(error_rationale) * (font_size + line_spacing)
        if isinstance(error_rationale, list)
        else 0
    )
    total_height = (
        y_offset
        + margin_y
        + metadata_height
        + question_height
        + options_height
        + explanation_height
        + error_rationale_height
    )

    # Resize surface if needed
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, int(total_height))
    context = cairo.Context(surface)
    context.set_font_size(font_size)

    # Set up background color
    context.set_source_rgba(*hex_to_rgb(background_color, opacity))
    context.rectangle(0, 0, width, total_height)
    context.fill()

    # Draw Metadata Tags
    if research_subject:
        # set bold text
        context.select_font_face(
            "Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD
        )
        context.set_source_rgba(*hex_to_rgb(subject_color, opacity))
        context.move_to(margin_x, y_offset)
        context.show_text(f"Subject:     {research_subject}")
        y_offset += font_size  # + line_spacing
        # unbold
        context.select_font_face(
            "Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL
        )

    if bloom_level:
        # set bold text
        context.select_font_face(
            "Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD
        )
        context.set_source_rgba(*hex_to_rgb(blooms_color, opacity))
        context.move_to(margin_x, y_offset)
        context.show_text(f"Bloom's:    {bloom_level}")
        y_offset += font_size  # + line_spacing
        # unbold
        context.select_font_face(
            "Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL
        )

    if specimen:
        # set bold text
        context.select_font_face(
            "Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD
        )
        context.set_source_rgba(*hex_to_rgb(specimen_color, opacity))
        context.move_to(margin_x, y_offset)
        context.show_text(f"Specimen: {specimen}")
        y_offset += font_size  # + line_spacing
        # unbold
        context.select_font_face(
            "Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL
        )

    # if error_category:
    #     # set bold text
    #     context.select_font_face(
    #         "Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD
    #     )
    #     context.set_source_rgba(*hex_to_rgb(error_category_color, opacity))
    #     context.move_to(margin_x, y_offset)
    #     context.show_text(f"Error type: {error_category.capitalize()}")
    #     y_offset += font_size  # + line_spacing
    #     # unbold
    #     context.select_font_face(
    #         "Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL
    #     )

    # Draw separator line between metadata and question
    # y_offset += 10
    context.set_source_rgb(0, 0, 0)
    context.set_line_width(1)
    context.move_to(margin_x, y_offset)
    context.line_to(width - margin_x, y_offset)
    context.stroke()
    y_offset += section_spacing

    # Draw Question
    context.set_source_rgba(*hex_to_rgb(question_color, opacity))
    for line in question_lines:
        context.move_to(margin_x, y_offset)
        context.show_text(line)
        y_offset += font_size  # + line_spacing

    # Draw Options
    pred_color: str = "#000000"
    prediction_coords: dict = None
    for idx, option_lines in enumerate(option_lines_wrapped):
        # Option background and border
        # box_height = (len(option_lines) * (font_size + line_spacing)) + line_height
        box_height = (len(option_lines) * (font_size)) + line_height
        if idx == correct_index:
            context.set_source_rgba(*hex_to_rgb(answer_background, opacity))
            context.set_line_width(1)
            context.rectangle(margin_x, y_offset, width - 2 * margin_x, box_height)
            context.fill_preserve()
            context.set_source_rgb(0, 0, 0)
            context.stroke()
        elif idx == prediction:
            context.set_source_rgba(*hex_to_rgb("#FFFFFF", opacity))
            context.set_line_width(1)
            context.rectangle(margin_x, y_offset, width - 2 * margin_x, box_height)
            context.fill_preserve()
            # add red X left of rectangle
            context.set_source_rgba(*hex_to_rgb("#FF0000", opacity))
            context.set_line_width(5)
            context.move_to(margin_x - 10, y_offset + 10)
            context.line_to(margin_x - 30, y_offset + 30)
            context.move_to(margin_x - 30, y_offset + 10)
            context.line_to(margin_x - 10, y_offset + 30)
            context.stroke()
            context.set_line_width(1)

        if idx == prediction:
            # color box border (all sides)
            pred_color = "#008000" if idx == correct_index else "#FF0000"  # 00FF00
            pred_box_height = (
                len(option_lines)
                * (font_size)
                # len(option_lines) * (font_size + line_spacing) # noqa
            ) + line_height
            pred_y_offset = y_offset
            # save prediction coords and draw at the end
            prediction_coords = (
                margin_x,
                pred_y_offset,
                width - 2 * margin_x,
                pred_box_height,
            )
            context.set_source_rgba(*hex_to_rgb(pred_color, opacity))
            context.set_line_width(1)
            context.rectangle(*prediction_coords)
            context.stroke()

        context.set_source_rgb(0, 0, 0)  # Border color
        context.stroke()

        # Option text
        context.set_source_rgba(*hex_to_rgb(option_color, opacity))
        for idx, line in enumerate(option_lines):
            if idx not in {correct_index, prediction}:
                continue

            # I added "+ (line_height)" to center text within the box. It looks okay
            # with font size 20, but may need adjustment to be generalizable.
            context.move_to(margin_x + 10, y_offset + line_height * 1.5)
            context.show_text(line)
            y_offset += font_size  # + line_spacing
        y_offset += 15

    # Draw prediction border to be on top
    if not prediction_coords:
        logger.error("Prediction coordinates not found.")
        return

    context.set_source_rgba(*hex_to_rgb(pred_color, opacity))
    context.set_line_width(2)
    context.rectangle(*prediction_coords)
    context.stroke()

    # Add space between options and explanation
    y_offset += section_spacing

    # Draw Explanation
    context.set_source_rgba(*hex_to_rgb(explanation_color, opacity))
    for idx, line in enumerate(explanation_lines):
        context.move_to(margin_x, y_offset)
        context.show_text(line)
        if len(explanation_lines) != idx:
            # only add line spacing if not the last line
            y_offset += font_size  # + line_spacing

    if error_category:
        # Draw separator line between explanation and error_category
        y_offset -= 10  #  line right after explanation
        context.set_source_rgb(0, 0, 0)
        context.set_line_width(1)
        context.move_to(margin_x, y_offset)
        context.line_to(width - margin_x, y_offset)
        context.stroke()
        y_offset += font_size

        # set bold text
        context.select_font_face(
            "Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD
        )
        context.set_source_rgba(*hex_to_rgb(error_category_color, opacity))
        context.move_to(margin_x, y_offset)
        context.show_text(f"Error type: {error_category.capitalize()}")
        y_offset += font_size  # + line_spacing
        # unbold
        context.select_font_face(
            "Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL
        )
    if isinstance(error_rationale, list):
        # set bold text
        context.select_font_face(
            "Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD
        )
        context.set_source_rgba(*hex_to_rgb(error_category_color, opacity))

        for idx, line in enumerate(error_rationale):
            context.move_to(margin_x, y_offset)
            context.show_text(line)
            if len(error_rationale) != idx:
                # only add line spacing if not the last line
                y_offset += font_size  # + line_spacing

        # unbold
        context.select_font_face(
            "Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL
        )

    # Save to PNG
    if not dry_run:
        surface.write_to_png(output_path)
        if error_category:
            output_path = output_path.parent.joinpath(error_category).joinpath(
                output_path.name
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # resize surface 2x
            surface.write_to_png(output_path)
            # PIL read image
            img = PIL.Image.open(output_path)
            # resize 2x
            img = img.resize((img.width * 4, img.height * 4))

            # read full image file
            disp_img = PIL.Image.open(image_filepath)

            # rotate image to be horizontal if rectangular
            if disp_img.height < disp_img.width:
                disp_img = disp_img.rotate(90, expand=True)

            # resize image to be 2x, if disp_img is <1/2 image size, resize to 2x
            if disp_img.width < img.width / 2:
                disp_img = disp_img.resize((disp_img.width * 2, disp_img.height * 2))

            # resize disp_img to have same height as img

            # extend canvase to right to include disp_img on right
            new_img = PIL.Image.new(
                "RGB",
                (img.width + disp_img.width, img.height),
                color=(255, 255, 255),
            )
            # add img left
            new_img.paste(img, (0, 0))
            # add disp_img right, center vertically
            new_img.paste(disp_img, (img.width, (img.height - disp_img.height) // 2))

            new_img.save(output_path)


@click.command()
@click.argument(
    "input-file", type=click.Path(dir_okay=False, exists=True, path_type=Path)
)
@click.option(
    "--output-dir", type=click.Path(file_okay=False, exists=False, path_type=Path)
)
@click.option(
    "--filter_col",
    default=None,
    type=dict,
    help="Filter to apply to the dataframe. {column_name: value}",
)
@click.option("--dry-run", is_flag=True, help="Perform a trial run with no changes.")
@click.version_option()
def main(
    input_file: Path,
    output_dir: Optional[Path] = None,
    filter_col: Optional[Dict[Any, Any]] = None,
    dry_run: bool = False,
) -> None:
    """Generate MCQ graphics from a CSV file."""
    output_dir = output_dir or input_file.parent.joinpath(input_file.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        LOG_DIR.joinpath(f"{Path(__file__).stem}.log"),
        rotation="10 MB",
        level="INFO",
    )

    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")

    if input_file.suffix == ".csv":
        df = pd.read_csv(input_file)
    elif input_file.suffix == ".xlsx":
        df = pd.read_excel(input_file)
    else:
        raise ValueError("Input file must be a CSV or Excel file.")

    # filter
    filter_col = {"correct": 0}
    if filter_col and isinstance(filter_col, dict):
        logger.info(f"Filtering dataframe with {filter_col}")
        for key, value in filter_col.items():
            if key in df.columns:
                df = df[df[key] == value]
            else:
                logger.warning(f"Column {key} not found in the dataframe.")

    logger.info(f"Loaded {len(df)} rows from input file.")
    question_key = "question_2"
    answer_key = "answer_2_formatted"
    choices_key = "choices_2"
    explanation_key = "msg"
    prediction_key: Optional[str] = "pred"
    metadata_cols = [
        "blooms_level",
        "use_case",
        "organism",
        "specimen",
        "research_subject",
    ]
    filter_options = True
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        key_image = int(row["key_image"])
        key_question = int(row["key_question"])
        question = row[question_key]
        answer = row[answer_key].strip()

        research_subject = "mcq"
        if "research_subject" in row and not pd.isna(row["research_subject"]):
            research_subject = " ".join(
                [x[:4] for x in row["research_subject"].split()]
            )
            research_subject = research_subject.replace(" ", "-").lower()

        use_case = row["use_case"] if "use_case" in row else "na"
        blooms_level = (
            int(row["blooms_level"])
            if "blooms_level" in row and not pd.isna(row["blooms_level"])
            else "na"
        )
        blooms_name = (
            f" ({row['blooms_name'].capitalize()})"
            if "blooms_name" in row and not pd.isna(row["blooms_name"])
            else ""
        )

        # get options
        options = eval(row[choices_key])
        options = [str(option).strip() for option in options]
        pred_index = row[prediction_key] if prediction_key else None
        correct_index = options.index(answer)
        if filter_options:
            options = [options[correct_index], options[pred_index]]
            correct_index = 0
            pred_index = 1

        pred_correct = "correct" if pred_index == correct_index else "incorrect"
        error_category = (
            row["error_category"]
            if "error_category" in row and not pd.isna(row["error_category"])
            else None
        )
        specimen = (
            row["specimen"]
            if "specimen" in row and not pd.isna(row["specimen"])
            else None
        )

        # set image path
        image_filepath = output_dir.joinpath("./images", f"{key_image:03d}.png")
        image_filepath = image_filepath if image_filepath.exists() else None

        # get explanation
        explanation = row[explanation_key].strip()
        abbrev_explanation = row["abbrev_msg"]
        if pd.isna(abbrev_explanation):
            continue

        output_file = f"{idx:03d}_{key_image:02d}_{key_question:02d}_{research_subject}_blooms-{blooms_level}_task-{use_case}_{pred_correct}.png"
        output_path = output_dir.joinpath(output_file)
        create_mcq_graphics(
            question,
            options,
            correct_index,
            # explanation,
            abbrev_explanation,
            output_path,
            prediction=pred_index,
            dry_run=dry_run,
            research_subject=row["research_subject"].capitalize(),
            bloom_level=f"Level {blooms_level}{blooms_name}",
            error_category=error_category,
            error_rationale=row["error_rationale"],
            specimen=specimen,
            image_filepath=image_filepath,
        )

    logger.info("Finished generating graphics.")


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    main()

# #     ##### Scratch filter and save pickle
# #     # collect list of items in error_category_list, exclude first elem
# #     error_category_list = df["error_category_list"].apply(lambda x: x.split(",")[1:])
# #     error_category_list = error_category_list.explode()
# #
# #     # temp
# #     # filter df to only those with non null error_category
# #     df = df[~df["error_category"].isnull()]
# #     df.reset_index(drop=True, inplace=True)
# #     # save new name
# #     output_file = output_dir.joinpath(f"{input_file.stem}_filter-error-category.csv")
# #     df.to_csv(output_file, index=False)
# #     df.to_excel(output_file.with_suffix(".xlsx"), index=False)
# #
#     # load pickle file with images
#     pkl_dir = output_dir
#     pkl_file = pkl_dir.joinpath(f"{input_file.stem}.pkl")
#
#     # load pick as binary
#     if pkl_file.exists():
#         images = pickle_reader(pkl_file)
#
#     # filter images to those with key_image in df
#     images = {
#         key: value for key, value in images.items() if key in df["key_image"].values
#     }
#
#     # save images to output_dir images
#     output_dir = output_dir.joinpath("all_images")
#     output_dir.mkdir(parents=True, exist_ok=True)
#     for key_image, img in images.items():
#
#         # convert to PIL
#         if len(img) > 1:
#             img_list = [PIL.Image.fromarray(img_) for img_ in img]
#             # combine into one image grid
#             for idx2, img in enumerate(img_list):
#                 img.save(output_dir.joinpath(f"{key_image:03d}_{idx2}.png"))
#         else:
#             img = PIL.Image.fromarray(img[0])
#
#             img.save(output_dir.joinpath(f"{key_image:03d}.png"))
# #
# #
# #     ##
#
# from PIL import Image
#
# def create_collage(images: list[str], collage_width: int, collage_height: int, grid_size: tuple[int, int]) -> Image:
#     """
#     Create a collage of images arranged in a grid.
#
#     Args:
#         images (list[str]): List of file paths to the images.
#         collage_width (int): Width of the collage in pixels.
#         collage_height (int): Height of the collage in pixels.
#         grid_size (tuple[int, int]): Number of rows and columns in the collage.
#
#     Returns:
#         Image: A PIL Image object of the collage.
#     """
#     rows, cols = grid_size
#     cell_width = collage_width // cols
#     cell_height = collage_height // rows
#
#     # Create a blank canvas
#     collage = Image.new('RGB', (collage_width, collage_height), 'white')
#
#     for index, img_path in enumerate(images):
#         if index >= rows * cols:
#             break
#
#         # Open and resize image
#         img = Image.open(img_path)
#         img = img.resize((cell_width, cell_height))
#
#         # Calculate position
#         row = index // cols
#         col = index % cols
#         x = col * cell_width
#         y = row * cell_height
#
#         # Paste image on canvas
#         collage.paste(img, (x, y))
#
#     return collage
#
#
# # Example usage # get all from output_dir
# image_paths = [ str(img) for img in output_dir.glob("*.png") ]
# # random shuffle
# import random
# random.shuffle(image_paths)
# # format for 8x11 page high res
# height = 2200
# width = 1600
# collage = create_collage(image_paths, width, height, (24, 16))
# collage.show()
# collage.save(output_dir.joinpath("collage_2.png"))
