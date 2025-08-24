// src/preprocess.c
// Reads CSV -> parallel means/stddev with OpenMP -> writes normalized CSV for sample_ files
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define MAX_LINE 4096
#define NUMF 8
static const char* FEAT[NUMF] = {
  "tempo","valence","energy","danceability","acousticness","popularity","year","duration_ms"
};

typedef struct {
  char id[64], name[128];
  double f[NUMF];
} Row;

static int find_col(const char* header, const char* key){
  char buf[MAX_LINE]; strncpy(buf, header, MAX_LINE-1); buf[MAX_LINE-1]='\0';
  int i=0; for(char* t=strtok(buf,","); t; t=strtok(NULL,","),++i){
    while(*t==' '||*t=='\t') t++;
    size_t n=strlen(t); while(n&& (t[n-1]=='\r'||t[n-1]=='\n'||t[n-1]==' '||t[n-1]=='\t')) t[--n]='\0';
    if(strcasecmp(t,key)==0) return i;
  } return -1;
}
static int parse_row(const char* line,int c_id,int c_name,int c_idx[NUMF],Row* r){
  char buf[MAX_LINE]; strncpy(buf,line,MAX_LINE-1); buf[MAX_LINE-1]='\0';
  int col=0, got_id=0, got_nm=0; char* s=NULL;
  for(char* t=strtok_r(buf,",",&s); t; t=strtok_r(NULL,",",&s),++col){
    while(*t==' '||*t=='\t') t++; size_t n=strlen(t);
    while(n&& (t[n-1]=='\r'||t[n-1]=='\n'||t[n-1]==' '||t[n-1]=='\t')) t[--n]='\0';
    if(col==c_id){ strncpy(r->id,t,sizeof(r->id)-1); r->id[sizeof(r->id)-1]='\0'; got_id=1; }
    if(col==c_name){ strncpy(r->name,t,sizeof(r->name)-1); r->name[sizeof(r->name)-1]='\0'; got_nm=1; }
    for(int k=0;k<NUMF;++k) if(col==c_idx[k]) r->f[k]=atof(t);
  }
  return got_id&&got_nm;
}

static int is_sample(const char* p){
  if(!p) return 0; const char* b=strrchr(p,'/'); 
#ifdef _WIN32
  if(!b) b=strrchr(p,'\\');
#endif
  b=b?b+1:p; return strncmp(b,"sample_",7)==0;
}

void preprocess_data(){
  const char* path=getenv("SPOTIFY_CSV");
  if(!path||!*path) path="data/sample_spotify.csv";
  FILE* fp=fopen(path,"r"); if(!fp){ fprintf(stderr,"[preprocess] open fail: %s\n",path); return; }

  char line[MAX_LINE];
  if(!fgets(line,sizeof(line),fp)){ fprintf(stderr,"[preprocess] empty CSV\n"); fclose(fp); return; }

  int c_id=find_col(line,"track_id"), c_name=find_col(line,"track_name"), c_idx[NUMF];
  for(int k=0;k<NUMF;++k) c_idx[k]=find_col(line,FEAT[k]);
  if(c_id<0||c_name<0){ fprintf(stderr,"[preprocess] missing id/name columns\n"); fclose(fp); return; }
  for(int k=0;k<NUMF;++k) if(c_idx[k]<0){ fprintf(stderr,"[preprocess] missing feature: %s\n",FEAT[k]); fclose(fp); return; }

size_t cap=1024,n=0; Row* rows=(Row*)malloc(cap*sizeof(Row)); if(!rows){ fclose(fp); return; }
  while(fgets(line,sizeof(line),fp)){
    if(n==cap){ cap*=2; Row* t=(Row*)realloc(rows,cap*sizeof(Row)); if(!t){ free(rows); fclose(fp); return; } rows=t; }
    if(parse_row(line,c_id,c_name,c_idx,&rows[n])) n++;
  }
  fclose(fp);
  if(!n){ fprintf(stderr,"[preprocess] no rows parsed\n"); free(rows); return; }

  printf("[preprocess] Loaded %zu rows from %s\n", n, path);
  printf("[preprocess] OpenMP threads: %d\n", omp_get_max_threads());

double sum[NUMF]={0};
  #pragma omp parallel for reduction(+:sum[:NUMF])
  for(long long i=0;i<(long long)n;++i) for(int k=0;k<NUMF;++k) sum[k]+=rows[i].f[k];
  double mean[NUMF]; for(int k=0;k<NUMF;++k) mean[k]=sum[k]/(double)n;

  double sse[NUMF]={0};
  #pragma omp parallel for reduction(+:sse[:NUMF])
  for(long long i=0;i<(long long)n;++i) for(int k=0;k<NUMF;++k){ double d=rows[i].f[k]-mean[k]; sse[k]+=d*d; }
  double sd[NUMF]; for(int k=0;k<NUMF;++k){ sd[k]=(n>1)?sqrt(sse[k]/(double)(n-1)):1.0; if(sd[k]==0) sd[k]=1.0; }

  printf("[preprocess] Means & StdDev:\n");
  for(int k=0;k<NUMF;++k) printf("  %-13s mean=%8.4f  sd=%8.4f\n", FEAT[k], mean[k], sd[k]);
 if(is_sample(path)){
    FILE* out=fopen("data/sample_spotify_normalized.csv","w");
    if(out){
      fprintf(out,"track_id,track_name");
      for(int k=0;k<NUMF;++k) fprintf(out,",%s_z",FEAT[k]); fprintf(out,"\n");

      #pragma omp parallel
      {
        char* buf=(char*)malloc(64*1024); size_t L=0,Cap=64*1024;
        #pragma omp for
        for(long long i=0;i<(long long)n;++i){
          char lineb[1024]; int len=snprintf(lineb,sizeof(lineb),"%s,%s",rows[i].id,rows[i].name);
          for(int k=0;k<NUMF;++k){ double z=(rows[i].f[k]-mean[k])/sd[k]; len+=snprintf(lineb+len,sizeof(lineb)-len,",%.6f",z); }
          len+=snprintf(lineb+len,sizeof(lineb)-len,"\n");
          if(L+len>=Cap){ Cap*=2; buf=(char*)realloc(buf,Cap); }
          memcpy(buf+L,lineb,(size_t)len); L+=len;
        }
        #pragma omp critical
        fwrite(buf,1,L,out);
        free(buf);
      }
      fclose(out);
      printf("[preprocess] Wrote data/sample_spotify_normalized.csv\n");
    }
  free(rows);
  printf("[preprocess] Done.\n");
}


