-- Column only. On Postgres 11+ a NOT NULL column with a constant default is a catalog change,
-- so this does not rewrite the table. The index change is the next migration.

-- AlterTable
ALTER TABLE "SteerOutput" ADD COLUMN     "presencePenalty" DOUBLE PRECISION NOT NULL DEFAULT 0;
