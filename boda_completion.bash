_boda()
{
  _boda_commands=$(boda compsup)
  local cur prev
  COMPREPLY=()
  #echo "COMP_WORDS ${COMP_WORDS[1]}"
  cur="${COMP_WORDS[COMP_CWORD]}"
  COMPREPLY=( $(compgen -W "${_boda_commands}" -- ${cur}) )
  return 0
}
complete -F _boda boda
