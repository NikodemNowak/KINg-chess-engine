BEGIN { FS = " \\| " }
{
  fen = $1; score = $2 + 0;
  n = split(fen, f, " "); fm = f[6] + 0;
  if (score < 0) score = -score;
  if (score <= 25 && fm >= 2 && fm <= 16) { print fen; c++; }
  if (c >= 6000) exit;
}
