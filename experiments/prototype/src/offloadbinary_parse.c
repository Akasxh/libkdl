/*
 * offloadbinary_parse.c -- Writer + parser for LLVM OffloadBinary format
 *
 * Implements the binary format defined in:
 *   llvm/include/llvm/Object/OffloadBinary.h
 *   llvm/lib/Object/OffloadBinary.cpp
 *
 * Format (little-endian, all fields uint64_t unless noted):
 *
 *   File header (48 bytes):
 *     [0]  Magic        = 0x10FF10AD
 *     [8]  Version      = 1
 *     [16] Size         = total file size in bytes
 *     [24] EntryOffset  = offset to first OffloadBinaryEntry
 *     [32] EntryCount   = number of entries
 *     [40] Padding      = 0
 *
 *   Each OffloadBinaryEntry (variable length):
 *     [0]  TheSize      = size of this entire entry (header + strings + image)
 *     [8]  ImageOffset  = offset from entry start to image bytes
 *     [16] ImageSize    = size of image bytes
 *     [24] StringOffset = offset from entry start to string table
 *     [32] StringSize   = size of string table (null-terminated key=value pairs)
 *     [40] ...strings (triple, arch, kind, ...)
 *     [...] image bytes
 *
 * Key strings are stored as a flat string table of null-terminated strings:
 *   "triple\0nvptx64-nvidia-cuda\0arch\0sm_75\0kind\0cuda\0"
 *
 * This standalone tool:
 *   1. Reads real .cubin files from a directory
 *   2. Packs them into a valid OffloadBinary file (/tmp/multi_arch.offloadbin)
 *   3. Parses the written file and validates each entry
 *   4. Prints all metadata (triple, arch, kind, image size)
 *
 * Build:
 *   gcc -O2 -Wall -std=c11 -o offloadbinary_parse offloadbinary_parse.c
 *
 * Usage:
 *   ./offloadbinary_parse /path/to/cubins/dir /tmp/output.offloadbin
 *   ./offloadbinary_parse --parse /tmp/output.offloadbin
 */

#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  OffloadBinary constants (from llvm/Object/OffloadBinary.h)        */
/* ------------------------------------------------------------------ */

#define OFFLOAD_BINARY_MAGIC    UINT64_C(0x10FF10AD)
#define OFFLOAD_BINARY_VERSION  UINT64_C(1)

/* Key string constants matching LLVM's OffloadBinary.cpp */
#define KEY_TRIPLE   "triple"
#define KEY_ARCH     "arch"
#define KEY_KIND     "kind"
#define KIND_CUDA    "cuda"
#define TRIPLE_NVPTX "nvptx64-nvidia-cuda"

#define MAX_ENTRIES    16
#define MAX_PATH_LEN   512
#define MAX_STRSIZE    256

/* ------------------------------------------------------------------ */
/*  On-disk structures (packed, little-endian)                        */
/* ------------------------------------------------------------------ */

typedef struct __attribute__((packed)) {
    uint64_t magic;
    uint64_t version;
    uint64_t size;           /* total file size */
    uint64_t entry_offset;   /* offset from file start to first entry */
    uint64_t entry_count;
    uint64_t padding;
} OffloadBinaryHeader;

typedef struct __attribute__((packed)) {
    uint64_t the_size;       /* total size of this entry */
    uint64_t image_offset;   /* offset from entry start to image bytes */
    uint64_t image_size;
    uint64_t string_offset;  /* offset from entry start to string table */
    uint64_t string_size;
} OffloadBinaryEntryHeader;

/* In-memory representation for building */
typedef struct {
    char     triple[64];
    char     arch[32];
    char     kind[32];
    uint8_t *image;
    uint64_t image_size;
} ImageEntry;

/* ------------------------------------------------------------------ */
/*  File I/O helpers                                                  */
/* ------------------------------------------------------------------ */

static uint8_t *read_file(const char *path, uint64_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    uint8_t *buf = malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *out_size = (uint64_t)sz;
    return buf;
}

/* Parse "sm_75" -> "sm_75" arch string from filename */
static void arch_from_filename(const char *name, char *out, int out_len) {
    const char *p = strstr(name, "sm");
    if (!p) { strncpy(out, "sm_75", out_len); return; }
    /* Copy "sm_XX" or "smXX" */
    const char *underscore = strchr(p, '_');
    if (underscore && (underscore - p) <= 2) {
        /* has underscore: sm_75 */
        snprintf(out, out_len, "sm_%s", underscore + 1);
        /* trim at first non-digit */
        char *end = out + 3;
        while (*end && (*end >= '0' && *end <= '9')) end++;
        *end = '\0';
    } else {
        /* no underscore: sm75 */
        snprintf(out, out_len, "sm_%s", p + 2);
        char *end = out + 3;
        while (*end && (*end >= '0' && *end <= '9')) end++;
        *end = '\0';
    }
}

/* ------------------------------------------------------------------ */
/*  Writer: pack entries into OffloadBinary file                      */
/* ------------------------------------------------------------------ */

static int write_offload_binary(const char *out_path,
                                 ImageEntry *entries, int n) {
    /*
     * Layout:
     *   OffloadBinaryHeader (48 bytes)
     *   [entry_0_header (40 bytes) | string_table | image_bytes]
     *   [entry_1_header (40 bytes) | string_table | image_bytes]
     *   ...
     */

    /* Pass 1: compute sizes */
    uint64_t entry_sizes[MAX_ENTRIES];
    uint64_t total = sizeof(OffloadBinaryHeader);

    for (int i = 0; i < n; i++) {
        ImageEntry *e = &entries[i];
        /* Build string table: "key\0value\0key\0value\0..." */
        uint64_t str_size = 0;
        str_size += strlen(KEY_TRIPLE) + 1 + strlen(e->triple) + 1;
        str_size += strlen(KEY_ARCH)   + 1 + strlen(e->arch)   + 1;
        str_size += strlen(KEY_KIND)   + 1 + strlen(e->kind)   + 1;

        uint64_t entry_hdr_size = sizeof(OffloadBinaryEntryHeader);
        entry_sizes[i] = entry_hdr_size + str_size + e->image_size;
        total += entry_sizes[i];
    }

    /* Allocate output buffer */
    uint8_t *buf = calloc(1, (size_t)total);
    if (!buf) { perror("calloc"); return -1; }

    /* Write file header */
    OffloadBinaryHeader *hdr = (OffloadBinaryHeader *)buf;
    hdr->magic        = OFFLOAD_BINARY_MAGIC;
    hdr->version      = OFFLOAD_BINARY_VERSION;
    hdr->size         = total;
    hdr->entry_offset = sizeof(OffloadBinaryHeader);
    hdr->entry_count  = (uint64_t)n;
    hdr->padding      = 0;

    /* Write each entry */
    uint64_t offset = sizeof(OffloadBinaryHeader);
    for (int i = 0; i < n; i++) {
        ImageEntry *e = &entries[i];
        uint8_t *entry_base = buf + offset;

        /* Build string table inline */
        uint8_t strtab[MAX_STRSIZE];
        uint64_t strtab_len = 0;

#define APPEND_STR(s) do { \
    size_t _l = strlen(s) + 1; \
    memcpy(strtab + strtab_len, s, _l); \
    strtab_len += _l; \
} while(0)

        APPEND_STR(KEY_TRIPLE); APPEND_STR(e->triple);
        APPEND_STR(KEY_ARCH);   APPEND_STR(e->arch);
        APPEND_STR(KEY_KIND);   APPEND_STR(e->kind);
#undef APPEND_STR

        uint64_t entry_hdr_size = sizeof(OffloadBinaryEntryHeader);
        uint64_t str_off   = entry_hdr_size;
        uint64_t img_off   = entry_hdr_size + strtab_len;

        /* Write entry header */
        OffloadBinaryEntryHeader *ehdr = (OffloadBinaryEntryHeader *)entry_base;
        ehdr->the_size     = entry_sizes[i];
        ehdr->image_offset = img_off;
        ehdr->image_size   = e->image_size;
        ehdr->string_offset = str_off;
        ehdr->string_size  = strtab_len;

        /* Write string table */
        memcpy(entry_base + str_off, strtab, strtab_len);

        /* Write image */
        memcpy(entry_base + img_off, e->image, e->image_size);

        offset += entry_sizes[i];
    }

    /* Write to file */
    FILE *f = fopen(out_path, "wb");
    if (!f) { perror(out_path); free(buf); return -1; }
    if (fwrite(buf, 1, (size_t)total, f) != (size_t)total) {
        perror("fwrite"); fclose(f); free(buf); return -1;
    }
    fclose(f);
    free(buf);

    printf("[offload_binary] wrote %lu bytes to %s (%d entries)\n",
           (unsigned long)total, out_path, n);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Parser: read and validate OffloadBinary file                      */
/* ------------------------------------------------------------------ */

static int parse_offload_binary(const char *path) {
    uint64_t file_size = 0;
    uint8_t *buf = read_file(path, &file_size);
    if (!buf) return -1;

    printf("\n=== Parsing OffloadBinary: %s (%lu bytes) ===\n",
           path, (unsigned long)file_size);

    /* Validate header */
    if (file_size < sizeof(OffloadBinaryHeader)) {
        fprintf(stderr, "File too small for header (%lu < %lu)\n",
                (unsigned long)file_size,
                (unsigned long)sizeof(OffloadBinaryHeader));
        free(buf);
        return -1;
    }

    OffloadBinaryHeader *hdr = (OffloadBinaryHeader *)buf;

    if (hdr->magic != OFFLOAD_BINARY_MAGIC) {
        fprintf(stderr, "Bad magic: 0x%lX (expected 0x%lX)\n",
                (unsigned long)hdr->magic,
                (unsigned long)OFFLOAD_BINARY_MAGIC);
        free(buf);
        return -1;
    }

    printf("  magic:        0x%lX (VALID)\n", (unsigned long)hdr->magic);
    printf("  version:      %lu\n", (unsigned long)hdr->version);
    printf("  total_size:   %lu bytes\n", (unsigned long)hdr->size);
    printf("  entry_offset: %lu\n", (unsigned long)hdr->entry_offset);
    printf("  entry_count:  %lu\n\n", (unsigned long)hdr->entry_count);

    if (hdr->size != file_size) {
        fprintf(stderr, "  WARNING: header.size (%lu) != actual (%lu)\n",
                (unsigned long)hdr->size, (unsigned long)file_size);
    }

    /* Walk entries */
    uint64_t entry_off = hdr->entry_offset;
    for (uint64_t i = 0; i < hdr->entry_count; i++) {
        if (entry_off + sizeof(OffloadBinaryEntryHeader) > file_size) {
            fprintf(stderr, "  Entry %lu out of bounds\n", (unsigned long)i);
            break;
        }

        OffloadBinaryEntryHeader *ehdr =
            (OffloadBinaryEntryHeader *)(buf + entry_off);

        printf("  Entry [%lu]:\n", (unsigned long)i);
        printf("    the_size:      %lu bytes\n", (unsigned long)ehdr->the_size);
        printf("    image_offset:  %lu\n", (unsigned long)ehdr->image_offset);
        printf("    image_size:    %lu bytes\n", (unsigned long)ehdr->image_size);
        printf("    string_offset: %lu\n", (unsigned long)ehdr->string_offset);
        printf("    string_size:   %lu bytes\n", (unsigned long)ehdr->string_size);

        /* Parse string table (key\0value\0 pairs) */
        const char *strtab = (const char *)(buf + entry_off + ehdr->string_offset);
        uint64_t strtab_end = ehdr->string_size;
        uint64_t pos = 0;
        printf("    metadata:\n");
        while (pos < strtab_end) {
            const char *key = strtab + pos;
            size_t klen = strlen(key);
            pos += klen + 1;
            if (pos >= strtab_end) break;
            const char *val = strtab + pos;
            size_t vlen = strlen(val);
            pos += vlen + 1;
            printf("      %s = %s\n", key, val);
        }

        /* Verify image bytes: check CUBIN magic (ELF: 0x7F 'E' 'L' 'F') */
        const uint8_t *img = buf + entry_off + ehdr->image_offset;
        if (ehdr->image_size >= 4) {
            int is_elf = (img[0] == 0x7F && img[1] == 'E' &&
                          img[2] == 'L' && img[3] == 'F');
            printf("    image_magic:   %02X %02X %02X %02X  (%s)\n",
                   img[0], img[1], img[2], img[3],
                   is_elf ? "ELF/CUBIN valid" : "unknown");
        }

        entry_off += ehdr->the_size;
        printf("\n");
    }

    free(buf);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Load cubins from directory into ImageEntry array                  */
/* ------------------------------------------------------------------ */

static int load_cubins_from_dir(const char *dir_path,
                                 ImageEntry *entries, int max) {
    DIR *d = opendir(dir_path);
    if (!d) { perror(dir_path); return -1; }

    int n = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL && n < max) {
        const char *name = ent->d_name;
        size_t len = strlen(name);
        if (len < 6 || strcmp(name + len - 6, ".cubin") != 0) continue;

        char path[MAX_PATH_LEN];
        snprintf(path, sizeof(path), "%s/%s", dir_path, name);

        uint64_t sz = 0;
        uint8_t *img = read_file(path, &sz);
        if (!img) continue;

        ImageEntry *e = &entries[n];
        strncpy(e->triple, TRIPLE_NVPTX, sizeof(e->triple) - 1);
        arch_from_filename(name, e->arch, sizeof(e->arch));
        strncpy(e->kind, KIND_CUDA, sizeof(e->kind) - 1);
        e->image = img;
        e->image_size = sz;

        printf("  loaded: %s -> arch=%s (%lu bytes)\n",
               name, e->arch, (unsigned long)sz);
        n++;
    }
    closedir(d);
    return n;
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    printf("=== OffloadBinary Writer + Parser (LLVM format) ===\n\n");

    /* Mode: --parse <file> */
    if (argc == 3 && strcmp(argv[1], "--parse") == 0) {
        return parse_offload_binary(argv[2]) == 0 ? 0 : 1;
    }

    /* Mode: <cubins_dir> <out_file> */
    if (argc < 3) {
        fprintf(stderr,
                "Usage:\n"
                "  %s <cubins_dir> <out.offloadbin>  -- pack cubins\n"
                "  %s --parse <out.offloadbin>        -- inspect file\n",
                argv[0], argv[0]);
        return 1;
    }

    const char *dir_path = argv[1];
    const char *out_path = argv[2];

    printf("[1] Loading cubins from: %s\n", dir_path);
    ImageEntry entries[MAX_ENTRIES];
    memset(entries, 0, sizeof(entries));

    int n = load_cubins_from_dir(dir_path, entries, MAX_ENTRIES);
    if (n <= 0) {
        fprintf(stderr, "No .cubin files found in %s\n", dir_path);
        return 1;
    }
    printf("  entries loaded: %d\n\n", n);

    printf("[2] Writing OffloadBinary: %s\n", out_path);
    int rc = write_offload_binary(out_path, entries, n);
    if (rc != 0) {
        fprintf(stderr, "Write failed\n");
        goto cleanup;
    }

    printf("\n[3] Parsing + validating written file\n");
    rc = parse_offload_binary(out_path);

cleanup:
    for (int i = 0; i < n; i++)
        if (entries[i].image) free(entries[i].image);

    return rc == 0 ? 0 : 1;
}
