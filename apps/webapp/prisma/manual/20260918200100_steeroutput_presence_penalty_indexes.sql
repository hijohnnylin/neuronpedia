-- Manual, lock-free version of migrations
--   20260918200000_steeroutput_presence_penalty          (the column)
--   20260918200100_steeroutput_presence_penalty_indexes  (the index rebuild)
--
-- The webapp build runs `prisma migrate deploy`, which applies every pending migration inside a
-- transaction, so the index rebuild would hold a lock on "SteerOutput" for the whole build. Run
-- this file BEFORE the branch deploys, then mark both migrations applied so deploy skips them.
--
-- From apps/webapp on this branch, with .env.prod present (it is gitignored; copy it in):
--
--   npx --no-install env-cmd -f .env.prod --use-shell \
--     'psql "$POSTGRES_URL_NON_POOLING" -v ON_ERROR_STOP=1 -f prisma/manual/20260918200100_steeroutput_presence_penalty_indexes.sql'
--
--   npx --no-install env-cmd -f .env.prod --use-shell \
--     'prisma migrate resolve --applied 20260918200000_steeroutput_presence_penalty && prisma migrate resolve --applied 20260918200100_steeroutput_presence_penalty_indexes'
--
-- Check, expect "No pending migrations to apply" and no drift:
--
--   npx --no-install env-cmd -f .env.prod --use-shell 'prisma migrate deploy'
--   npx --no-install env-cmd -f .env.prod --use-shell \
--     'prisma migrate diff --from-url "$POSTGRES_URL_NON_POOLING" --to-schema-datamodel prisma/schema.prisma'
--
-- `migrate resolve` only writes rows to _prisma_migrations. psql runs each statement in its own
-- transaction, which CONCURRENTLY needs, so this file must go through psql, not a Prisma migration.
--
-- The new indexes build under temporary names while the old ones keep serving lookups. The final
-- swap is one short transaction of renames. If a CONCURRENTLY build fails part way it leaves an
-- INVALID index behind; drop it and rerun this file:
--   DROP INDEX CONCURRENTLY IF EXISTS "steerIndex_new";

-- Column. A NOT NULL column with a constant default is a catalog change on Postgres 11+.
ALTER TABLE "SteerOutput" ADD COLUMN IF NOT EXISTS "presencePenalty" DOUBLE PRECISION NOT NULL DEFAULT 0;

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
