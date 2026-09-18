-- Manual, lock-free version of migration 20260918200100_steeroutput_presence_penalty_indexes.
--
-- Run before `prisma migrate deploy` (the webapp build runs it), on the direct connection, not the
-- pooler. psql runs each statement in its own transaction, which CONCURRENTLY needs:
--
--   psql "$POSTGRES_URL_NON_POOLING" -v ON_ERROR_STOP=1 -f prisma/manual/20260918200100_steeroutput_presence_penalty_indexes.sql
--
-- Then tell Prisma the migration is done, so deploy skips its locking copy:
--
--   npx prisma migrate resolve --applied 20260918200100_steeroutput_presence_penalty_indexes
--
-- Needs migration 20260918200000_steeroutput_presence_penalty (the column) applied first.
--
-- The new indexes build under temporary names while the old ones keep serving lookups. The final
-- swap is one short transaction of renames.
--
-- If a CONCURRENTLY build fails part way it leaves an INVALID index behind. Drop it and rerun:
--   DROP INDEX CONCURRENTLY IF EXISTS "steerIndex_new";

CREATE INDEX CONCURRENTLY IF NOT EXISTS "steerIndex_new" ON "SteerOutput"("modelId", "type", "inputTextMd5", "temperature", "numTokens", "presencePenalty", "seed", "strengthMultiplier", "version", "steerSpecialTokens", "steerMethod");

CREATE INDEX CONCURRENTLY IF NOT EXISTS "steerIndex2_new" ON "SteerOutput"("modelId", "type", "inputTextChatTemplateMd5", "temperature", "numTokens", "presencePenalty", "seed", "strengthMultiplier", "version", "steerSpecialTokens", "steerMethod");

CREATE INDEX CONCURRENTLY IF NOT EXISTS "steerIndexWithoutType_new" ON "SteerOutput"("modelId", "inputTextMd5", "temperature", "numTokens", "presencePenalty", "seed", "strengthMultiplier", "version", "steerSpecialTokens", "steerMethod");

CREATE INDEX CONCURRENTLY IF NOT EXISTS "steerIndexWithoutType2_new" ON "SteerOutput"("modelId", "inputTextChatTemplateMd5", "temperature", "numTokens", "presencePenalty", "seed", "strengthMultiplier", "version", "steerSpecialTokens", "steerMethod");

-- Swap. DROP INDEX and ALTER INDEX RENAME take a brief exclusive lock on the table.
BEGIN;
DROP INDEX "steerIndex";
DROP INDEX "steerIndex2";
DROP INDEX "steerIndexWithoutType";
DROP INDEX "steerIndexWithoutType2";
ALTER INDEX "steerIndex_new" RENAME TO "steerIndex";
ALTER INDEX "steerIndex2_new" RENAME TO "steerIndex2";
ALTER INDEX "steerIndexWithoutType_new" RENAME TO "steerIndexWithoutType";
ALTER INDEX "steerIndexWithoutType2_new" RENAME TO "steerIndexWithoutType2";
COMMIT;
