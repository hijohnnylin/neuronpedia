-- The squashed migration creates "Explanation_embedding_idx" as a btree. pgvector
-- cannot use a btree to answer a `<=>` nearest-neighbour query, so every database
-- built from the migrations sequential-scans Explanation on semantic search while
-- production, whose index was rebuilt by hand as HNSW, does not. Settings here
-- match production exactly: m = 16, ef_construction = 100.
--
-- Guarded rather than unconditional. Building an HNSW index takes the table's
-- write lock for its duration, which on production-sized Explanation is not
-- something a deploy should do silently. Where the index is already HNSW this
-- does nothing at all.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = current_schema()
          AND indexname = 'Explanation_embedding_idx'
          AND indexdef ILIKE '%hnsw%'
    ) THEN
        RAISE NOTICE 'Explanation_embedding_idx is already HNSW; leaving it alone.';
    ELSE
        DROP INDEX IF EXISTS "Explanation_embedding_idx";
        CREATE INDEX "Explanation_embedding_idx"
            ON "Explanation" USING hnsw ("embedding" vector_cosine_ops)
            WITH (m = 16, ef_construction = 100);
    END IF;
END $$;
