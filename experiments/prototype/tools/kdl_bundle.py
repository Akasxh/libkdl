#!/usr/bin/env python3
"""kdl_bundle.py — Create Multi-Target Bundle (MTB) files for mlir-hetero-dispatch.

MTB format (see ARCHITECTURE.md §4.1):
  [Header 32B][Kernel Table][Variant Table][String Table][Binary Data]

Usage:
  python kdl_bundle.py \\
    --kernel matmul \\
    --variant nvptx:sm_80:matmul.cubin:'{"target":"nvptx","min_arch":"sm_80"}':matmul_kernel \\
    --variant x86:x86-64-v3:matmul.o:'{"target":"x86"}':matmul_kernel \\
    -o matmul.mtb

  python kdl_bundle.py --demo -o demo.mtb
"""

import argparse
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

MAGIC = b"KDL_MTB\x00"
FORMAT_VERSION = 1

# target_kind encoding (§4.1 Variant Table +0)
TARGET_KIND = {
    "nvptx":  0,
    "amdgcn": 1,
    "spirv":  2,
    "x86":    3,
}

HEADER_FMT   = "<8sIIIIII"   # magic(8) ver num_k num_v strtab_off bindat_off reserved
HEADER_SIZE  = struct.calcsize(HEADER_FMT)   # 32

KERNEL_FMT   = "<III"        # name_off first_variant_idx num_variants
KERNEL_SIZE  = struct.calcsize(KERNEL_FMT)   # 12

VARIANT_FMT  = "<IIIIQQIi"  # target_kind chip_off contract_off priority
                              # binary_offset(u64) binary_size(u64) ep_off reserved
VARIANT_SIZE = struct.calcsize(VARIANT_FMT)  # 40

assert HEADER_SIZE  == 32, HEADER_SIZE
assert KERNEL_SIZE  == 12, KERNEL_SIZE
assert VARIANT_SIZE == 40, VARIANT_SIZE

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class VariantSpec:
    target_kind: int       # encoded TARGET_KIND value
    chip: str              # e.g. "sm_80", "gfx942", "x86-64-v3"
    binary_path: Path
    contract: str          # JSON string
    entry_point: str
    priority: int = 0

@dataclass
class KernelSpec:
    name: str
    variants: list[VariantSpec] = field(default_factory=list)

# ── String table helper ───────────────────────────────────────────────────────

class StringTable:
    """Append-only table of NUL-terminated UTF-8 strings with deduplication."""

    def __init__(self) -> None:
        self._data = bytearray()
        self._index: dict[str, int] = {}

    def intern(self, s: str) -> int:
        if s in self._index:
            return self._index[s]
        offset = len(self._data)
        self._index[s] = offset
        self._data.extend(s.encode("utf-8"))
        self._data.append(0)
        return offset

    def bytes(self) -> bytes:
        return bytes(self._data)

# ── Bundle writer ─────────────────────────────────────────────────────────────

def write_bundle(kernels: list[KernelSpec], out_path: Path) -> None:
    strtab = StringTable()
    binary_chunks: list[bytes] = []

    # First pass: collect all strings and binary blobs so we can compute offsets.
    total_variants = sum(len(k.variants) for k in kernels)

    kernel_table_offset  = HEADER_SIZE
    variant_table_offset = kernel_table_offset + len(kernels) * KERNEL_SIZE
    # string_table_offset and binary_data_offset resolved after first pass

    # Intern all strings up front so the table is complete before we write.
    for k in kernels:
        strtab.intern(k.name)
        for v in k.variants:
            strtab.intern(v.chip)
            strtab.intern(v.contract)
            strtab.intern(v.entry_point)

    strtab_bytes = strtab.bytes()
    string_table_offset = variant_table_offset + total_variants * VARIANT_SIZE
    binary_data_offset  = string_table_offset + len(strtab_bytes)

    # Second pass: load binaries and record their offsets.
    binary_offset_map: list[tuple[int, int]] = []  # (offset, size) per variant in order
    cursor = 0
    for k in kernels:
        for v in k.variants:
            data = v.binary_path.read_bytes()
            binary_chunks.append(data)
            binary_offset_map.append((cursor, len(data)))
            cursor += len(data)

    # Assemble output.
    out = bytearray()

    # Header
    out += struct.pack(
        HEADER_FMT,
        MAGIC,
        FORMAT_VERSION,
        len(kernels),
        total_variants,
        string_table_offset,
        binary_data_offset,
        0,  # reserved
    )

    # Kernel table
    variant_idx = 0
    for k in kernels:
        out += struct.pack(
            KERNEL_FMT,
            strtab._index[k.name],
            variant_idx,
            len(k.variants),
        )
        variant_idx += len(k.variants)

    # Variant table
    blob_idx = 0
    for k in kernels:
        for v in k.variants:
            bin_off, bin_sz = binary_offset_map[blob_idx]
            out += struct.pack(
                VARIANT_FMT,
                v.target_kind,
                strtab._index[v.chip],
                strtab._index[v.contract],
                v.priority,
                binary_data_offset + bin_off,
                bin_sz,
                strtab._index[v.entry_point],
                0,  # reserved
            )
            blob_idx += 1

    # String table
    out += strtab_bytes

    # Binary data
    for chunk in binary_chunks:
        out += chunk

    out_path.write_bytes(bytes(out))
    print(f"Wrote {out_path}  ({len(out):,} bytes)")
    print(f"  kernels:  {len(kernels)}")
    print(f"  variants: {total_variants}")
    print(f"  strtab:   {len(strtab_bytes)} bytes")
    print(f"  bindata:  {sum(len(c) for c in binary_chunks)} bytes")

# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_variant(spec: str, priority: int = 0) -> tuple[str, VariantSpec]:
    """Parse  TARGET:CHIP:FILE:CONTRACT_JSON:ENTRY_POINT  (colon-delimited, 5 fields).

    CONTRACT_JSON may itself contain colons, so we split on the first 4 colons only.
    """
    parts = spec.split(":", 4)
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            f"--variant must be TARGET:CHIP:FILE:CONTRACT_JSON:ENTRY_POINT, got: {spec!r}"
        )
    target_str, chip, file_str, contract_json, entry_point = parts

    target_str_lower = target_str.lower()
    if target_str_lower not in TARGET_KIND:
        raise argparse.ArgumentTypeError(
            f"Unknown target {target_str!r}. Valid: {list(TARGET_KIND)}"
        )

    # Validate JSON
    try:
        json.loads(contract_json)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid contract JSON: {exc}") from exc

    binary_path = Path(file_str)
    if not binary_path.exists():
        raise argparse.ArgumentTypeError(f"Binary file not found: {binary_path}")

    return (
        target_str_lower,
        VariantSpec(
            target_kind=TARGET_KIND[target_str_lower],
            chip=chip,
            binary_path=binary_path,
            contract=contract_json,
            entry_point=entry_point,
            priority=priority,
        ),
    )

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--kernel",
        metavar="NAME",
        action="append",
        dest="kernel_names",
        help="Kernel name (repeat for multiple kernels); variants are assigned "
             "to the most recently declared kernel.",
    )
    p.add_argument(
        "--variant",
        metavar="TARGET:CHIP:FILE:CONTRACT:ENTRY",
        action="append",
        dest="variant_specs",
        help="Variant specification (repeat for multiple variants).",
    )
    p.add_argument("-o", "--output", required=True, metavar="FILE", help="Output MTB path.")
    p.add_argument(
        "--demo",
        action="store_true",
        help="Create a synthetic demo bundle (no real binaries required).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print layout details.",
    )
    return p

# ── Demo bundle ───────────────────────────────────────────────────────────────

def make_demo_bundle(out_path: Path) -> None:
    """Create a demo MTB with synthetic binary blobs for testing."""
    import tempfile, os

    demo_kernels: list[KernelSpec] = []

    kernel_defs = [
        {
            "name": "matmul",
            "variants": [
                {
                    "target": "nvptx",
                    "chip": "sm_80",
                    "contract": json.dumps({
                        "target": "nvptx",
                        "min_arch": "sm_80",
                        "min_shared_mem_kb": 48,
                    }),
                    "entry": "matmul_kernel",
                    "payload": b"\x7fELF" + b"\xde\xad\xbe\xef" * 16 + b"NVPTX-SM80-CUBIN",
                    "priority": 0,
                },
                {
                    "target": "amdgcn",
                    "chip": "gfx942",
                    "contract": json.dumps({
                        "target": "amdgcn",
                        "min_arch": "gfx942",
                    }),
                    "entry": "matmul_kernel",
                    "payload": b"HSA\x00" + b"\xca\xfe\xba\xbe" * 16 + b"AMDGCN-GFX942-HSACO",
                    "priority": 1,
                },
                {
                    "target": "x86",
                    "chip": "x86-64-v3",
                    "contract": json.dumps({
                        "target": "x86",
                        "min_arch": "x86-64-v3",
                    }),
                    "entry": "matmul_kernel",
                    "payload": b"\x7fELF" + b"\x00\x01\x02\x03" * 16 + b"X86-64-V3-OBJ",
                    "priority": 10,
                },
            ],
        },
        {
            "name": "relu",
            "variants": [
                {
                    "target": "nvptx",
                    "chip": "sm_80",
                    "contract": json.dumps({"target": "nvptx", "min_arch": "sm_80"}),
                    "entry": "relu_kernel",
                    "payload": b"\x7fELF" + b"\xaa\xbb\xcc\xdd" * 8 + b"RELU-NVPTX",
                    "priority": 0,
                },
                {
                    "target": "x86",
                    "chip": "x86-64-v2",
                    "contract": json.dumps({"target": "x86", "min_arch": "x86-64-v2"}),
                    "entry": "relu_kernel",
                    "payload": b"\x7fELF" + b"\x11\x22\x33\x44" * 8 + b"RELU-X86",
                    "priority": 5,
                },
            ],
        },
    ]

    tmpfiles: list[str] = []
    try:
        for kdef in kernel_defs:
            kspec = KernelSpec(name=kdef["name"])
            for v in kdef["variants"]:
                fd, tmp = tempfile.mkstemp(suffix=".bin", prefix=f"kdl_{kdef['name']}_")
                tmpfiles.append(tmp)
                os.write(fd, v["payload"])
                os.close(fd)
                kspec.variants.append(VariantSpec(
                    target_kind=TARGET_KIND[v["target"]],
                    chip=v["chip"],
                    binary_path=Path(tmp),
                    contract=v["contract"],
                    entry_point=v["entry"],
                    priority=v["priority"],
                ))
            demo_kernels.append(kspec)

        write_bundle(demo_kernels, out_path)
    finally:
        for tmp in tmpfiles:
            try:
                os.unlink(tmp)
            except OSError:
                pass

# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    out_path = Path(args.output)

    if args.demo:
        make_demo_bundle(out_path)
        return 0

    if not args.kernel_names:
        parser.error("Specify at least one --kernel NAME, or use --demo.")
    if not args.variant_specs:
        parser.error("Specify at least one --variant, or use --demo.")

    # Map variants to their kernels (variants assigned to the most-recently-seen kernel).
    kernels: list[KernelSpec] = [KernelSpec(name=n) for n in args.kernel_names]

    for i, spec_str in enumerate(args.variant_specs):
        try:
            _, vspec = parse_variant(spec_str, priority=i)
        except argparse.ArgumentTypeError as exc:
            print(f"Error in --variant #{i+1}: {exc}", file=sys.stderr)
            return 1
        # Assign to last kernel (simple heuristic; multi-kernel workflows use wrapper scripts).
        kernels[-1].variants.append(vspec)

    if any(not k.variants for k in kernels):
        empty = [k.name for k in kernels if not k.variants]
        print(f"Error: kernels with no variants: {empty}", file=sys.stderr)
        return 1

    write_bundle(kernels, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
