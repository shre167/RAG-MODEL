For the chunking itself, I actually think it's mostly fine.

The important part is this:

chunk_id = f"{filename}_{section_index}_{idx}"


and

RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)


Those are reasonable choices.

Why JWST became 1 chunk

The log:

James Webb Space Telescope (JWST).txt → 1 new chunks


does not automatically mean a bug.

It just means:

len(text_with_heading) <= CHUNK_SIZE


or only one segment was produced after splitting.

Add a temporary log:

logger.info(
    "%s length=%d chars -> %d chunks",
    filename,
    len(text),
    len(sec_chunks),
)


inside chunk_documents().

Then you'll see:

JWST.txt length=523 chars -> 1 chunks
Mars.txt length=8421 chars -> 14 chunks


which makes things much easier to validate.

What I'd improve
1. Make chunk IDs independent of filename

Currently:

chunk_id = f"{filename}_{section_index}_{idx}"


Then later:

_candidate_key(
    filename,
    chunk_id,
)


becomes:

JWST.txt::chunk-JWST.txt_0_0


The filename is duplicated.

I'd simplify to:

chunk_id = f"{section_index}_{idx}"


because the filename is already part of the final key.

2. Add chunk statistics after ingestion

After:

chunks = chunk_documents(...)


log:

from collections import Counter

counts = Counter(
    chunk["filename"]
    for chunk in chunks
)

for fname, count in counts.items():
    logger.info(
        "%s -> %d chunks",
        fname,
        count,
    )


Then you'd immediately spot:

JWST.txt -> 1 chunks
Mars.txt -> 17 chunks
Saturn.txt -> 22 chunks


instead of guessing.

3. Check your actual settings

The biggest factor is still:

CHUNK_SIZE
CHUNK_OVERLAP


If you currently have:

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


that's pretty normal for a RAG.

If it's:

CHUNK_SIZE = 2000


or

CHUNK_SIZE = 4000


then one-chunk documents become very common.

One thing that could cause problems later

Your IDs depend on:

section_index
idx


If someone inserts a new heading near the beginning of a document, all later section numbers may shift:

old:
0_0
1_0
2_0

new:
0_0
1_0
2_0
3_0


That can make chunk identities unstable.

A more robust approach is:

hash(filename + chunk_text)


for chunk IDs.

But that's an enhancement, not a bug.

My overall assessment

Chunker: ✅ mostly good
 Heading detection: ✅ good
 Chunk IDs: ⚠️ could be more robust
 JWST generating 1 chunk: ✅ likely normal unless the file is large
 Main issue in your system: still appears to be synchronization between Chroma, BM25, and deleted files, not the chunking algorithm itself.
