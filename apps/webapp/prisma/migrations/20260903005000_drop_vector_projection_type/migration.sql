-- The second half of the expand/contract begun in 20260902230000. That migration left
-- `projectionType` in place because a deploy applies migrations at build time, so the previous
-- revision kept serving -- and reading the column -- until the build finished.
--
-- The revision that reads it is gone now: `VectorTag` rows say what a vector is, and nothing gates
-- a read on the answer. The enum goes with the column, which is its only user.
ALTER TABLE "Vector" DROP COLUMN "projectionType";

DROP TYPE "ProjectionType";
