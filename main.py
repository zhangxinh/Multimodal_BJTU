import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List
import shutil

from src import config
from src.embeddings import TextEmbedder
from src.image_manager import ImageManager
from src.paper_manager import PaperManager


def parse_topics(raw: str) -> List[str]:
    return [t.strip() for t in raw.split(",") if t.strip()]


def prepare_output_dir(command: str) -> Path:
    """在 output/ 下按时间戳+命令创建唯一的输出目录。"""
    base_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    candidate = config.OUTPUT_DIR / f"{base_ts}_{command}"
    suffix = 1
    while candidate.exists():
        candidate = config.OUTPUT_DIR / f"{base_ts}_{command}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def write_json(out_dir: Path, filename: str, data: dict | list) -> None:
    path = out_dir / filename
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(out_dir: Path, filename: str, content: str) -> None:
    path = out_dir / filename
    path.write_text(content, encoding="utf-8")


def announce(out_dir: Path) -> None:
    # 命令行只显示输出目录
    print(f"输出目录: {out_dir}")


def cmd_add_paper(args, paper_mgr: PaperManager, raw_cmd: str) -> None:
    topics = parse_topics(args.topics)
    result = paper_mgr.add_paper(args.path, topics)
    out_dir = prepare_output_dir("add_paper")
    write_json(out_dir, "result.json", {"command": raw_cmd, "result": result})
    announce(out_dir)


def cmd_search_paper(args, paper_mgr: PaperManager, raw_cmd: str) -> None:
    results = paper_mgr.search_papers(args.query, top_k=args.top_k)
    out_dir = prepare_output_dir("search_paper")
    lines: List[str] = [f"命令: {raw_cmd}", f"查询: {args.query}", ""]
    if not results:
        lines.append("未找到已索引论文，请先添加。")
    else:
        for item in results:
            if args.files_only:
                lines.append(item["path"])
            else:
                topics = ", ".join(item.get("topics", []))
                line = f"[{item['score']:.3f}] {item['path']} ({topics})"
                lines.append(line)
                if item.get("summary"):
                    preview = item["summary"][:200].replace("\n", " ")
                    lines.append(f"    {preview}")
    write_text(out_dir, "results.txt", "\n".join(lines))
    announce(out_dir)


def cmd_search_chunk(args, paper_mgr: PaperManager, raw_cmd: str) -> None:
    results = paper_mgr.search_chunks(args.query, top_k=args.top_k)
    out_dir = prepare_output_dir("search_chunk")
    lines: List[str] = [f"命令: {raw_cmd}", f"查询: {args.query}", ""]
    if not results:
        lines.append("未找到片段索引，请先添加论文。")
    else:
        for item in results:
            preview = item["text"][:220].replace("\n", " ")
            lines.append(f"[{item['score']:.3f}] {item['paper_path']}#page{item['page']}: {preview}")
    write_text(out_dir, "results.txt", "\n".join(lines))
    announce(out_dir)


def cmd_search_image(args, image_mgr: ImageManager, raw_cmd: str) -> None:
    results = image_mgr.search_images(args.query, top_k=args.top_k)
    out_dir = prepare_output_dir("search_image")
    lines: List[str] = [f"命令: {raw_cmd}", f"查询: {args.query}", ""]
    if not results:
        lines.append("images/ 下没有可用图片索引，请先放入图片。")
        write_text(out_dir, "results.txt", "\n".join(lines))
        announce(out_dir)
        return

    best_path_written = None
    for idx, item in enumerate(results):
        preview = item["caption"][:160].replace("\n", " ")
        line = f"[{item['score']:.3f}] {item['path']} :: {preview}"
        lines.append(line)
        if idx == 0:
            src = Path(item["path"])
            if src.exists():
                dest = out_dir / src.name
                if dest.exists():
                    stem, suffix = dest.stem, dest.suffix
                    n = 1
                    while True:
                        candidate = out_dir / f"{stem}_{n}{suffix}"
                        if not candidate.exists():
                            dest = candidate
                            break
                        n += 1
                shutil.copy2(src, dest)
                best_path_written = dest
    if best_path_written:
        lines.append("")
        lines.append(f"最相关的图片已拷贝到: {best_path_written}")
    write_text(out_dir, "results.txt", "\n".join(lines))
    announce(out_dir)


def cmd_organize(args, paper_mgr: PaperManager, raw_cmd: str) -> None:
    topics = parse_topics(args.topics)
    results = paper_mgr.batch_organize(args.folder, topics)
    out_dir = prepare_output_dir("organize_papers")
    write_json(out_dir, "result.json", {"command": raw_cmd, "result": results})
    announce(out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="论文与图片的多模态管理工具")
    subparsers = parser.add_subparsers(dest="command")

    add_paper = subparsers.add_parser("add_paper", help="添加并分类单篇论文")
    add_paper.add_argument("path", help="PDF 文件路径")
    add_paper.add_argument("--topics", default="", help="候选主题，逗号分隔")

    search_paper = subparsers.add_parser("search_paper", help="论文语义检索")
    search_paper.add_argument("query")
    search_paper.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    search_paper.add_argument(
        "--files-only", action="store_true", help="仅输出匹配的文件路径"
    )

    search_chunk = subparsers.add_parser("search_chunk", help="检索并返回论文片段")
    search_chunk.add_argument("query")
    search_chunk.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)

    search_image = subparsers.add_parser("search_image", help="以文搜图")
    search_image.add_argument("query")
    search_image.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)

    organize = subparsers.add_parser("organize_papers", help="整理指定目录下的 PDF")
    organize.add_argument("folder")
    organize.add_argument("--topics", default="", help="候选主题，逗号分隔")

    sort_paper = subparsers.add_parser(
        "sort_paper",
        help="一键整理 papers/ 下所有 PDF 到主题子目录",
    )
    sort_paper.add_argument(
        "--topics",
        default="",
        help="候选主题，逗号分隔（可选）",
    )

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    raw_args = argv if argv is not None else sys.argv[1:]
    raw_cmd = f"python main.py {' '.join(raw_args)}"
    args = parser.parse_args(raw_args)
    if not args.command:
        parser.print_help()
        sys.exit(1)

    embedder = TextEmbedder()
    paper_mgr = PaperManager(embedder=embedder)
    image_mgr = ImageManager(embedder=embedder)

    if args.command == "add_paper":
        cmd_add_paper(args, paper_mgr, raw_cmd)
    elif args.command == "search_paper":
        cmd_search_paper(args, paper_mgr, raw_cmd)
    elif args.command == "search_chunk":
        cmd_search_chunk(args, paper_mgr, raw_cmd)
    elif args.command == "search_image":
        cmd_search_image(args, image_mgr, raw_cmd)
    elif args.command == "organize_papers":
        cmd_organize(args, paper_mgr, raw_cmd)
    elif args.command == "sort_paper":
        # 默认整理 papers 根目录
        args.folder = str(config.PAPERS_DIR)
        cmd_organize(args, paper_mgr, raw_cmd)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
