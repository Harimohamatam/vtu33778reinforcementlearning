import checkpointData from "./alignment-checkpoints.json";

type LanguageModelParameters = {
  E: number[][];
  pos: number[][];
  W1: number[][];
  b1: number[];
  W2: number[][];
  b2: number[];
};

type RewardModelParameters = {
  E: number[][];
  U1: number[][];
  c1: number[];
  u2: number[];
  c2: number[];
};

type Checkpoint = {
  vocab: string[];
  maxLen: number;
  bos: number;
  eos: number;
  pad: number;
  context: number;
  models: Record<AlignmentMethod, LanguageModelParameters>;
  rewardModel: RewardModelParameters;
  training: {
    sftEpochs: number;
    rlhfBeta: number;
    rlhfIterations: number;
    dpoBeta: number;
    dpoEpochs: number;
  };
};

const checkpoint = checkpointData as Checkpoint;
const wordToId = new Map(checkpoint.vocab.map((word, id) => [word, id]));
const lowercaseWordToId = new Map(
  checkpoint.vocab.map((word, id) => [word.toLowerCase(), id]),
);

const politeWords = new Set(["please", "thanks", "welcome", "understand", "help"]);
const actionWords = new Set([
  "check",
  "reset",
  "review",
  "update",
  "track",
  "contact",
  "explain",
  "try",
]);
const unhelpfulWords = new Set(["ignore", "whatever", "obviously", "never", "impossible"]);
const functionWords = new Set([
  "i",
  "can",
  "will",
  "your",
  "the",
  "a",
  "to",
  "for",
  "with",
  "in",
  "on",
  "if",
  "we",
  "let",
  "me",
  "this",
  "you",
  "are",
  "is",
  "it",
  "and",
  "before",
  "using",
]);

export type AlignmentMethod = "sft" | "rlhf" | "dpo";

export type LiveEvaluation = {
  goldReward: number;
  learnedReward: number;
  klDrift: number;
  independentJudge: number;
};

export type GeneratedResponse = {
  method: AlignmentMethod;
  text: string;
  metrics: LiveEvaluation;
};

export type PromptCheck = { tokenIds: number[]; unsupported: string[] };

function normalizeWord(word: string) {
  return word.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "");
}

export function checkPrompt(prompt: string): PromptCheck {
  const tokenIds: number[] = [];
  const unsupported: string[] = [];

  for (const rawWord of prompt.trim().split(/\s+/).filter(Boolean)) {
    const word = normalizeWord(rawWord);
    const id = wordToId.get(word) ?? lowercaseWordToId.get(word.toLowerCase());
    if (!word || id === undefined) unsupported.push(rawWord);
    else tokenIds.push(id);
  }

  return { tokenIds, unsupported: [...new Set(unsupported)] };
}

function nextTokenLogProbabilities(
  parameters: LanguageModelParameters,
  sequence: number[],
  tokenPosition: number,
) {
  const context = Array.from(
    { length: checkpoint.context },
    (_, index) => {
      const sequenceIndex = sequence.length - checkpoint.context + index;
      return sequenceIndex < 0 ? checkpoint.bos : sequence[sequenceIndex] ?? checkpoint.bos;
    },
  );
  const features = context.flatMap((tokenId) => parameters.E[tokenId] ?? []);
  features.push(...(parameters.pos[tokenPosition] ?? []));

  const hidden = (parameters.b1 ?? []).map((bias, index) => {
    const weights = parameters.W1.map((row) => row[index] ?? 0);
    return Math.tanh(
      features.reduce((sum, feature, featureIndex) => sum + feature * (weights[featureIndex] ?? 0), bias),
    );
  });

  const logits = parameters.b2.map((bias, tokenId) => {
    if (tokenId === checkpoint.pad) return Number.NEGATIVE_INFINITY;
    const weights = parameters.W2.map((row) => row[tokenId] ?? 0);
    return hidden.reduce((sum, feature, index) => sum + feature * (weights[index] ?? 0), bias);
  });
  const maximum = Math.max(...logits);
  const logNormalizer = Math.log(logits.reduce((sum, value) => sum + Math.exp(value - maximum), 0));
  return logits.map((value) => value - maximum - logNormalizer);
}

function sampleNextToken(logProbabilities: number[]) {
  const draw = Math.random();
  let cumulative = 0;
  for (let id = 0; id < logProbabilities.length; id += 1) {
    cumulative += Math.exp(logProbabilities[id] ?? Number.NEGATIVE_INFINITY);
    if (draw < cumulative) return id;
  }
  return checkpoint.eos;
}

function generateTokens(method: AlignmentMethod, prefix: number[]) {
  const sequence = [checkpoint.bos, ...prefix];
  const model = checkpoint.models[method];

  while (sequence.length < checkpoint.maxLen) {
    if (sequence.length === checkpoint.maxLen - 1) {
      sequence.push(checkpoint.eos);
      break;
    }

    const logProbabilities = nextTokenLogProbabilities(model, sequence, sequence.length - 1);
    const next = sampleNextToken(logProbabilities);
    sequence.push(next);
    if (next === checkpoint.eos) break;
  }

  if (sequence.at(-1) !== checkpoint.eos) sequence.push(checkpoint.eos);
  return sequence;
}

function sequenceLogProbability(method: AlignmentMethod, sequence: number[]) {
  const model = checkpoint.models[method];
  let sum = 0;

  for (let index = 1; index < sequence.length; index += 1) {
    const tokenId = sequence[index];
    if (tokenId === undefined || tokenId === checkpoint.pad) break;
    const logProbabilities = nextTokenLogProbabilities(model, sequence.slice(0, index), index - 1);
    sum += logProbabilities[tokenId] ?? 0;
    if (tokenId === checkpoint.eos) break;
  }

  return sum;
}

function goldReward(text: string) {
  const words = text.toLowerCase().split(/\s+/).filter(Boolean);
  if (words.length === 0) return -2;

  const contentWords = words.filter((word) => !functionWords.has(word));
  const uniqueContentWords = new Set(contentWords);
  const firstWord = words[0] ?? "";
  return (
    0.55 * [...politeWords].filter((word) => words.includes(word)).length +
    0.65 * [...actionWords].filter((word) => words.includes(word)).length -
    0.85 * [...unhelpfulWords].filter((word) => words.includes(word)).length -
    0.12 * Math.max(0, words.length - 9) -
    0.9 * Math.max(0, contentWords.length - uniqueContentWords.size) -
    (new Set(["i", "thanks", "let", "we"]).has(firstWord) ? 0 : 0.3)
  );
}

function learnedReward(sequence: number[]) {
  const model = checkpoint.rewardModel;
  const contentIds = sequence.filter((id) => id > checkpoint.eos);
  const count = Math.max(1, contentIds.length);
  const meanEmbedding = (model.E[0] ?? []).map(
    (_, dimension) =>
      contentIds.reduce((sum, id) => sum + (model.E[id]?.[dimension] ?? 0), 0) / count,
  );
  const features = [...meanEmbedding, count / 10];
  const hidden = model.c1.map((bias, index) =>
    Math.tanh(
      features.reduce(
        (sum, feature, featureIndex) =>
          sum + feature * (model.U1[featureIndex]?.[index] ?? 0),
        bias,
      ),
    ),
  );
  return hidden.reduce((sum, value, index) => sum + value * (model.u2[index] ?? 0), model.c2[0] ?? 0);
}

function independentJudge(text: string) {
  const words = text.toLowerCase().split(/\s+/).filter(Boolean);
  if (words.length === 0) return -3;
  let score = 0;
  if (words.some((word) => new Set(["help", "explain", "check", "reset", "review", "update", "track", "contact"]).has(word))) score += 1;
  if (words.some((word) => new Set(["please", "thanks", "understand", "welcome"]).has(word))) score += 0.7;
  if (words.length <= 10) score += 0.4;
  if (words.length - new Set(words).size > 0) score -= 0.8;
  if (words.some((word) => new Set(["ignore", "whatever", "impossible"]).has(word))) score -= 1;
  return score;
}

function evaluate(method: AlignmentMethod, sequence: number[], text: string): LiveEvaluation {
  return {
    goldReward: goldReward(text),
    learnedReward: learnedReward(sequence),
    klDrift: sequenceLogProbability(method, sequence) - sequenceLogProbability("sft", sequence),
    independentJudge: independentJudge(text),
  };
}

function decode(sequence: number[]) {
  return sequence
    .filter((id) => id !== checkpoint.bos && id !== checkpoint.eos && id !== checkpoint.pad)
    .map((id) => checkpoint.vocab[id] ?? "")
    .filter(Boolean)
    .join(" ");
}

export function generateResponses(prompt: string, methods: AlignmentMethod[]): GeneratedResponse[] {
  const { tokenIds, unsupported } = checkPrompt(prompt);
  if (unsupported.length > 0) {
    throw new Error(`Outside this model’s vocabulary: ${unsupported.join(", ")}`);
  }
  if (tokenIds.length === 0) throw new Error("Enter at least one word to continue.");
  if (tokenIds.length > checkpoint.maxLen - 2) {
    throw new Error(`Use ${checkpoint.maxLen - 2} words or fewer so the model has room to continue.`);
  }

  return methods.map((method) => {
    const sequence = generateTokens(method, tokenIds);
    const text = decode(sequence);
    return { method, text, metrics: evaluate(method, sequence, text) };
  });
}

export const modelProvenance = checkpoint.training;