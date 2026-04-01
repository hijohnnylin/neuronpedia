-- AlterTable
ALTER TABLE "ProblemNode" ADD COLUMN     "author" TEXT,
ADD COLUMN     "nodeTypes" TEXT[] DEFAULT ARRAY['topic']::TEXT[];
