_boda()
{
  IFS=$'\n'
  _compsup_params=($(boda compsup "${COMP_WORDS[@]}"))
  unset IFS
  _compopt_params=()
  _compgen_params=()
  last_was_o=0
  for param in "${_compsup_params[@]}"; do 
      if [ "$param" = "-o" ]; then last_was_o=2; fi
      if [ $last_was_o -ne 0 ]; then
	  _compopt_params+=("${param}")
	  let last_was_o--
      else
	  _compgen_params+=("${param}") 
      fi
  done
  #printf -- 'param:%s\n' "${_compgen_params[@]}"
  COMPREPLY=()
  COMPREPLY=($(compgen "${_compgen_params[@]}"))
  if [ ${#_compopt_params[@]} -ne 0 ]; then compopt "${_compopt_params[@]}"; fi
  return 0
}
complete -F _boda boda
