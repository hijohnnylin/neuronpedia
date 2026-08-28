-- Contract half of the ComputeHost migration.
--
-- 20260826190000_add_compute_host copied these tables into ComputeHost and left
-- them in place, so that the release running at the time kept serving and a
-- rollback stayed safe. Both of those windows have closed: the registry is live
-- and resolving every route the old tables did. This drops the originals.
--
-- Irreversible in the sense that matters: once these are gone, the pre-registry
-- release can no longer be rolled back to, because it reads them and would find
-- nothing. The data itself is not lost -- ComputeHost holds the same host URLs
-- and links -- but recovering the old shape would mean restoring from a backup.

DROP TABLE IF EXISTS "InferenceHostSourceOnSource";
DROP TABLE IF EXISTS "GraphHostSourceOnSourceSet";
DROP TABLE IF EXISTS "InferenceHostSource";
DROP TABLE IF EXISTS "GraphHostSource";

-- Only ever used by InferenceHostSource.engine. The registry does not record an
-- engine: the server picks its own backend at startup and reports it through
-- /capabilities, so a routing decision was being made on a value nothing set.
DROP TYPE IF EXISTS "InferenceEngine";

-- Superseded by ComputeHost rows with service = 'NLA', one per NlaSource.
ALTER TABLE "NlaSource" DROP COLUMN IF EXISTS "servers";
