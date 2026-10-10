import { createFileRoute } from "@tanstack/react-router";
import { ArrowRight, FlaskConical, RotateCcw } from "lucide-react";
import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  checkPrompt,
  generateResponses,
  modelProvenance,
  type AlignmentMethod,
  type GeneratedResponse,
} from "@/lib/alignment";

type ViewMode = AlignmentMethod | "compare";

const methodNames: Record<AlignmentMethod, string> = {
  sft: "SFT",
  rlhf: "RLHF",
  dpo: "DPO",
};

const metricNames = [
  { key: "goldReward", label: "Gold rubric" },
  { key: "learnedReward", label: "Learned preference" },
  { key: "klDrift", label: "KL drift vs SFT" },
  { key: "independentJudge", label: "Independent judge" },
] as const;

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Campus Help-Desk Alignment Lab" },
      {
        name: "description",
        content: "Generate and compare responses from reproduced SFT, RLHF, and DPO course-project models.",
      },
      { property: "og:title", content: "Campus Help-Desk Alignment Lab" },
      {
        property: "og:description",
        content: "An interactive comparison of SFT, RLHF, and DPO using the course notebook’s compact language model.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: AlignmentLab,
});

function AlignmentLab() {
  const [mode, setMode] = useState<ViewMode>("sft");
  const [prompt, setPrompt] = useState("I can help");
  const [responses, setResponses] = useState<GeneratedResponse[]>([]);
  const [error, setError] = useState("");
  const [hasGenerated, setHasGenerated] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const promptCheck = checkPrompt(prompt);

  function runExperiment() {
    setError("");
    setIsGenerating(true);
    try {
      const selectedMethods: AlignmentMethod[] =
        mode === "compare" ? ["sft", "rlhf", "dpo"] : [mode];
      const nextResponses = generateResponses(prompt, selectedMethods);
      setResponses(nextResponses);
      setHasGenerated(true);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to generate a response.");
      setResponses([]);
      setHasGenerated(false);
    } finally {
      setIsGenerating(false);
    }
  }

  function changeMode(value: string) {
    if (value === "sft" || value === "rlhf" || value === "dpo" || value === "compare") {
      setMode(value);
      setResponses([]);
      setHasGenerated(false);
    }
  }

  function clearResults() {
    setResponses([]);
    setHasGenerated(false);
    setError("");
  }

  return (
    <main className="min-h-screen bg-background text-foreground">
      <div className="mx-auto max-w-5xl px-5 pb-16 pt-7 sm:px-8 sm:pt-10">
        <header className="flex items-center justify-between border-b border-border pb-5">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground">
              <FlaskConical aria-hidden="true" className="size-5" />
            </div>
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold">Alignment lab</p>
              <p className="text-xs text-muted-foreground">Campus help desk · course project</p>
            </div>
          </div>
          <Badge variant="outline" className="hidden shrink-0 sm:inline-flex">
            Notebook model
          </Badge>
        </header>

        <section className="pb-8 pt-9 sm:pb-10 sm:pt-12">
          <div className="max-w-2xl">
            <p className="mb-2 text-sm font-medium text-primary">SFT · RLHF · DPO</p>
            <h1 className="text-3xl font-semibold leading-tight sm:text-4xl">
              Compare aligned responses.
            </h1>
            <p className="mt-3 max-w-xl text-sm leading-6 text-muted-foreground sm:text-base">
              Continue a short help-desk prompt with the notebook’s small, synthetic language model.
            </p>
          </div>
        </section>

        <section aria-label="Response generator" className="border-y border-border py-6 sm:py-7">
          <div className="flex flex-col gap-5">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-sm font-medium">Alignment method</p>
              <Tabs value={mode} onValueChange={changeMode}>
                <TabsList className="grid h-10 w-full grid-cols-4 rounded-lg sm:w-auto">
                  <TabsTrigger value="sft">SFT</TabsTrigger>
                  <TabsTrigger value="rlhf">RLHF</TabsTrigger>
                  <TabsTrigger value="dpo">DPO</TabsTrigger>
                  <TabsTrigger value="compare">Compare</TabsTrigger>
                </TabsList>
              </Tabs>
            </div>

            <label className="block text-sm font-medium" htmlFor="prompt">
              Text prompt
              <Textarea
                id="prompt"
                value={prompt}
                onChange={(event) => {
                  setPrompt(event.target.value);
                  clearResults();
                }}
                onKeyDown={(event) => {
                  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") runExperiment();
                }}
                placeholder="I can help"
                rows={2}
                aria-describedby="prompt-description prompt-validation"
                className="mt-2 min-h-24 resize-y rounded-lg border-border bg-card text-base shadow-none placeholder:text-muted-foreground/70"
              />
            </label>

            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="min-w-0 text-xs leading-5 text-muted-foreground">
                <p id="prompt-description">
                  A starting phrase, not a general chat model. Use words from its small training vocabulary.
                </p>
                <p id="prompt-validation" className={error ? "mt-1 text-destructive" : "mt-1"} role={error ? "alert" : undefined}>
                  {error || (promptCheck.unsupported.length > 0
                    ? `Not in vocabulary: ${promptCheck.unsupported.join(", ")}`
                    : `${promptCheck.tokenIds.length} vocabulary word${promptCheck.tokenIds.length === 1 ? "" : "s"}`)}
                </p>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                {hasGenerated && (
                  <Button variant="ghost" size="icon" onClick={clearResults} aria-label="Clear responses" title="Clear responses">
                    <RotateCcw aria-hidden="true" />
                  </Button>
                )}
                <Button onClick={runExperiment} disabled={isGenerating || promptCheck.unsupported.length > 0}>
                  {isGenerating ? "Generating…" : mode === "compare" ? "Compare all three" : "Generate response"}
                  <ArrowRight aria-hidden="true" />
                </Button>
              </div>
            </div>
          </div>
        </section>

        <section aria-live="polite" aria-label="Generated responses" className="pt-7 sm:pt-9">
          {responses.length > 0 ? (
            <div className={responses.length === 3 ? "grid gap-4 lg:grid-cols-3" : "grid gap-4"}>
              {responses.map((response) => (
                <ResponsePanel key={response.method} response={response} />
              ))}
            </div>
          ) : (
            <div className="flex min-h-32 flex-col justify-center border-b border-border pb-7">
              <p className="text-sm font-medium">
                {error ? "Prompt not supported" : "Your response will appear here"}
              </p>
              <p className="mt-1 text-sm text-muted-foreground">
                {error
                  ? "Try a shorter phrase using words from the notebook’s vocabulary."
                  : "Choose an alignment method and generate a fresh response to see its live scores."}
              </p>
            </div>
          )}
        </section>

        <footer className="mt-8 border-t border-border pt-5">
          <p className="text-xs font-medium text-foreground">Reproduced from the supplied notebook</p>
          <p className="mt-1 max-w-3xl text-xs leading-5 text-muted-foreground">
            The uploaded notebook included training code, not saved model weights. These checkpoints were retrained from that code; they are not the original saved models. The learned scores below are evaluated on each generated response.
          </p>
          <p className="mt-3 text-xs text-muted-foreground">
            SFT {modelProvenance.sftEpochs} epochs · RLHF β {modelProvenance.rlhfBeta.toFixed(2)}, {modelProvenance.rlhfIterations} steps · DPO β {modelProvenance.dpoBeta.toFixed(2)}, {modelProvenance.dpoEpochs} epochs
          </p>
        </footer>
      </div>
    </main>
  );
}

function ResponsePanel({ response }: { response: GeneratedResponse }) {
  return (
    <article className="flex min-w-0 flex-col rounded-lg border border-border bg-card p-4 sm:p-5">
      <div className="flex items-center justify-between gap-3 border-b border-border pb-3">
        <h2 className="text-sm font-semibold">{methodNames[response.method]}</h2>
        <Badge variant="secondary" className="shrink-0 font-normal">
          Live evaluation
        </Badge>
      </div>
      <p className="min-h-20 py-4 text-base leading-7 text-foreground">{response.text || "No continuation was sampled."}</p>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-4 border-t border-border pt-4">
        {metricNames.map(({ key, label }) => (
          <div className="min-w-0" key={key}>
            <dt className="truncate text-xs leading-5 text-muted-foreground">{label}</dt>
            <dd className="mt-0.5 text-lg font-semibold tabular-nums">
              {response.metrics[key].toFixed(2)}
            </dd>
          </div>
        ))}
      </dl>
    </article>
  );
}